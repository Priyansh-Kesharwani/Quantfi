import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class PurgedKFold:

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not 0 <= embargo_pct < 0.5:
            raise ValueError("embargo_pct must be in [0, 0.5)")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def get_n_splits(self) -> int:
        return self.n_splits

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        groups: np.ndarray = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X) if hasattr(X, '__len__') else X
        embargo_size = int(n_samples * self.embargo_pct)
        indices = np.arange(n_samples)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[:n_samples % self.n_splits] += 1
        fold_starts = np.cumsum(np.r_[0, fold_sizes])

        for i in range(self.n_splits):
            test_start = fold_starts[i]
            test_end = fold_starts[i + 1]
            test_idx = indices[test_start:test_end]

            embargo_end = min(test_end + embargo_size, n_samples)

            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_start:embargo_end] = False
            train_idx = indices[train_mask]

            if len(train_idx) == 0:
                logger.warning(f"Fold {i}: empty training set after purge+embargo")
                continue

            yield train_idx, test_idx


class WalkForwardCV:

    def __init__(
        self,
        n_splits: int = 5,
        min_train_pct: float = 0.3,
        embargo_pct: float = 0.01,
        expanding: bool = True
    ):
        self.n_splits = n_splits
        self.min_train_pct = min_train_pct
        self.embargo_pct = embargo_pct
        self.expanding = expanding

    def get_n_splits(self) -> int:
        return self.n_splits

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        groups: np.ndarray = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X) if hasattr(X, '__len__') else X
        embargo_size = int(n_samples * self.embargo_pct)
        min_train = int(n_samples * self.min_train_pct)
        indices = np.arange(n_samples)

        oos_size = n_samples - min_train
        if oos_size <= 0:
            raise ValueError("min_train_pct too large; no OOS data")

        test_window = oos_size // self.n_splits
        if test_window < 1:
            raise ValueError("Not enough OOS data for requested n_splits")

        for i in range(self.n_splits):
            test_start = min_train + i * test_window
            test_end = min(test_start + test_window, n_samples)
            test_idx = indices[test_start:test_end]

            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, test_start - min_train)

            train_end = max(0, test_start - embargo_size)
            train_idx = indices[train_start:train_end]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx


class BlockBootstrap:

    def __init__(
        self,
        n_bootstraps: int = 100,
        block_length: Optional[int] = None,
        seed: int = 42
    ):
        self.n_bootstraps = n_bootstraps
        self.block_length = block_length
        self.seed = seed

    def resample(self, data: np.ndarray) -> Iterator[np.ndarray]:
        rng = np.random.RandomState(self.seed)
        n = len(data)
        avg_block = self.block_length or max(1, int(np.sqrt(n)))
        p_break = 1.0 / avg_block                                     

        for _ in range(self.n_bootstraps):
            indices = np.empty(n, dtype=int)
            pos = rng.randint(0, n)

            for t in range(n):
                if rng.random() < p_break and t > 0:
                    pos = rng.randint(0, n)
                indices[t] = pos % n                 
                pos += 1

            yield data[indices]

    def confidence_interval(
        self,
        data: np.ndarray,
        statistic_fn,
        alpha: float = 0.05
    ) -> Tuple[float, float, float]:
        point = statistic_fn(data)
        boot_stats = np.array([
            statistic_fn(sample) for sample in self.resample(data)
        ])

        lower = np.percentile(boot_stats, 100 * alpha / 2)
        upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

        return float(point), float(lower), float(upper)


@dataclass
class FoldResult:
    fold_idx: int
    train_size: int
    test_size: int
    train_dates: Optional[Tuple[str, str]] = None
    test_dates: Optional[Tuple[str, str]] = None
    dca_cost_improvement_pct: float = 0.0
    decile_monotonicity: Dict[str, float] = field(default_factory=dict)
    mean_score: float = 50.0
    std_score: float = 0.0
    sharpe_like: float = 0.0
    warnings: List[str] = field(default_factory=list)


def walk_forward_validate(
    df: pd.DataFrame,
    symbol: str,
    scorer_config: Optional[Any] = None,
    n_splits: int = 5,
    embargo_pct: float = 0.02,
    method: str = "purged_kfold"
) -> List[FoldResult]:
    from score_engine import CompositeScorer, ScorerConfig
    from backtester.diagnostics import DiagnosticBacktester, BacktestConfig

    if scorer_config is None:
        scorer_config = ScorerConfig(r_thresh=0.5, S_scale=1.5)

    if method == "purged_kfold":
        cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    elif method == "walk_forward":
        cv = WalkForwardCV(
            n_splits=n_splits, min_train_pct=0.3, embargo_pct=embargo_pct
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(df)):
        logger.info(
            f"  Fold {fold_i}: train={len(train_idx)}, "
            f"test={len(test_idx)}, embargo={embargo_pct}"
        )

        try:
            test_df = df.iloc[test_idx].copy()
            scorer = CompositeScorer(scorer_config)
            score_result = scorer.fit_transform(test_df)

            bt_config = BacktestConfig(
                forward_windows=[5, 10, 20],
                n_quantiles=5,                                             
                dca_frequency=5,
                transform_timeout=30
            )
            backtester = DiagnosticBacktester(bt_config)
            bt_result = backtester.run_backtest(
                symbol=f"{symbol}_fold{fold_i}",
                scores=score_result.scores,
                prices=test_df['Close'].values,
                dates=test_df.index
            )

            valid_scores = score_result.scores[~np.isnan(score_result.scores)]
            mono = {}
            for k, da in bt_result.decile_analysis.items():
                mono[k] = da.monotonicity

            fold_result = FoldResult(
                fold_idx=fold_i,
                train_size=len(train_idx),
                test_size=len(test_idx),
                train_dates=(
                    str(df.index[train_idx[0]]),
                    str(df.index[train_idx[-1]])
                ),
                test_dates=(
                    str(df.index[test_idx[0]]),
                    str(df.index[test_idx[-1]])
                ),
                dca_cost_improvement_pct=(
                    bt_result.dca_comparison.cost_improvement_pct
                    if bt_result.dca_comparison else 0.0
                ),
                decile_monotonicity=mono,
                mean_score=float(np.mean(valid_scores)) if len(valid_scores) > 0 else 50.0,
                std_score=float(np.std(valid_scores)) if len(valid_scores) > 0 else 0.0,
                warnings=bt_result.warnings + bt_result.errors
            )

        except Exception as e:
            logger.error(f"  Fold {fold_i} failed: {e}")
            fold_result = FoldResult(
                fold_idx=fold_i,
                train_size=len(train_idx),
                test_size=len(test_idx),
                warnings=[str(e)]
            )

        fold_results.append(fold_result)

    if fold_results:
        mean_scores = [f.mean_score for f in fold_results]
        cost_improvements = [f.dca_cost_improvement_pct for f in fold_results]
        logger.info(
            f"\n  Walk-forward summary for {symbol}:"
            f"\n    Folds: {len(fold_results)}"
            f"\n    Mean score across folds: {np.mean(mean_scores):.1f} ± {np.std(mean_scores):.1f}"
            f"\n    DCA cost improvement: {np.mean(cost_improvements):.2f}% ± {np.std(cost_improvements):.2f}%"
        )

    return fold_results
