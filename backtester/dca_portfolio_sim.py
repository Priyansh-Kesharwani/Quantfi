import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from dataclasses import dataclass, field
import logging
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DCAPortfolioConfig:

    base_investment: float = 1000.0

    boost_threshold: float = 60.0
    boost_multiplier: float = 1.5

    aggressive_threshold: float = 75.0
    aggressive_multiplier: float = 2.0

    cadence_days: int = 5           

    max_single_allocation: float = 10000.0                             
    max_total_budget: Optional[float] = None                                 

    transaction_cost_pct: float = 0.001          

    slippage_pct: float = 0.0005         

    forward_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])

    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_investment": self.base_investment,
            "boost_threshold": self.boost_threshold,
            "boost_multiplier": self.boost_multiplier,
            "aggressive_threshold": self.aggressive_threshold,
            "aggressive_multiplier": self.aggressive_multiplier,
            "cadence_days": self.cadence_days,
            "transaction_cost_pct": self.transaction_cost_pct,
            "slippage_pct": self.slippage_pct,
        }


@dataclass
class DCATransaction:
    date: datetime
    price: float
    investment: float
    units: float
    tier: str                                             
    score: float
    cost_with_fees: float
    cumulative_invested: float
    cumulative_units: float


class PortfolioSimResult(NamedTuple):
    total_invested: float
    total_units: float
    avg_cost_basis: float
    final_value: float
    total_return_pct: float
    total_fees: float

    tier_summary: Dict[str, Dict[str, float]]

    transactions: List[Dict[str, Any]]
    equity_curve: pd.DataFrame

    vs_uniform: Dict[str, float]                             

    config: Dict[str, Any]
    meta: Dict[str, Any]


class DCAPortfolioSimulator:

    def __init__(self, config: Optional[DCAPortfolioConfig] = None):
        self.config = config or DCAPortfolioConfig()

    def run(
        self,
        scores: np.ndarray,
        prices: np.ndarray,
        dates: pd.DatetimeIndex,
        symbol: str = "UNKNOWN",
    ) -> PortfolioSimResult:
        cfg = self.config
        n = len(prices)

        buy_indices = list(range(0, n, cfg.cadence_days))

        transactions: List[DCATransaction] = []
        cumulative_invested = 0.0
        cumulative_units = 0.0
        total_fees = 0.0
        tier_counts = {"base": 0, "boost": 0, "aggressive": 0}
        tier_amounts = {"base": 0.0, "boost": 0.0, "aggressive": 0.0}

        for idx in buy_indices:
            price = prices[idx]
            score = scores[idx]

            if np.isnan(price) or price <= 0:
                continue

            if np.isnan(score):
                tier = "base"
                investment = cfg.base_investment
            elif score >= cfg.aggressive_threshold:
                tier = "aggressive"
                investment = cfg.base_investment * cfg.aggressive_multiplier
            elif score >= cfg.boost_threshold:
                tier = "boost"
                investment = cfg.base_investment * cfg.boost_multiplier
            else:
                tier = "base"
                investment = cfg.base_investment

            investment = min(investment, cfg.max_single_allocation)

            if cfg.max_total_budget is not None:
                remaining = cfg.max_total_budget - cumulative_invested
                if remaining <= 0:
                    break
                investment = min(investment, remaining)

            effective_price = price * (1 + cfg.slippage_pct)
            fees = investment * cfg.transaction_cost_pct
            net_investment = investment - fees

            units = net_investment / effective_price

            cumulative_invested += investment
            cumulative_units += units
            total_fees += fees
            tier_counts[tier] += 1
            tier_amounts[tier] += investment

            transactions.append(DCATransaction(
                date=dates[idx] if idx < len(dates) else datetime.utcnow(),
                price=float(price),
                investment=float(investment),
                units=float(units),
                tier=tier,
                score=float(score) if not np.isnan(score) else -1.0,
                cost_with_fees=float(investment),
                cumulative_invested=float(cumulative_invested),
                cumulative_units=float(cumulative_units),
            ))

        final_price = prices[-1] if not np.isnan(prices[-1]) else prices[~np.isnan(prices)][-1]
        final_value = cumulative_units * final_price
        avg_cost = cumulative_invested / cumulative_units if cumulative_units > 0 else 0
        total_return_pct = (
            (final_value - cumulative_invested) / cumulative_invested * 100
            if cumulative_invested > 0 else 0
        )

        total_buys = sum(tier_counts.values())
        tier_summary = {}
        for tier_name in ["base", "boost", "aggressive"]:
            tier_summary[tier_name] = {
                "count": tier_counts[tier_name],
                "pct_of_buys": tier_counts[tier_name] / total_buys * 100 if total_buys > 0 else 0,
                "total_invested": tier_amounts[tier_name],
                "pct_of_capital": tier_amounts[tier_name] / cumulative_invested * 100 if cumulative_invested > 0 else 0,
            }

        equity_curve = self._build_equity_curve(transactions, prices, dates)

        vs_uniform = self._compare_vs_uniform(
            scores, prices, dates, cumulative_invested, total_buys
        )

        txn_dicts = [
            {
                "date": str(t.date),
                "price": t.price,
                "investment": t.investment,
                "units": t.units,
                "tier": t.tier,
                "score": t.score,
            }
            for t in transactions
        ]

        return PortfolioSimResult(
            total_invested=float(cumulative_invested),
            total_units=float(cumulative_units),
            avg_cost_basis=float(avg_cost),
            final_value=float(final_value),
            total_return_pct=float(total_return_pct),
            total_fees=float(total_fees),
            tier_summary=tier_summary,
            transactions=txn_dicts,
            equity_curve=equity_curve,
            vs_uniform=vs_uniform,
            config=self.config.to_dict(),
            meta={
                "symbol": symbol,
                "n_observations": n,
                "n_buy_days": total_buys,
                "date_range": f"{dates[0]} → {dates[-1]}" if len(dates) > 0 else "N/A",
                "computed_at": datetime.utcnow().isoformat(),
            },
        )

    def _build_equity_curve(
        self,
        transactions: List[DCATransaction],
        prices: np.ndarray,
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        n = len(prices)
        cum_units = np.zeros(n)
        cum_invested = np.zeros(n)

        txn_idx = 0
        running_units = 0.0
        running_invested = 0.0

        for i in range(n):
            if txn_idx < len(transactions):
                txn = transactions[txn_idx]
                txn_date = pd.Timestamp(txn.date)
                if i < len(dates) and dates[i] >= txn_date:
                    running_units += txn.units
                    running_invested += txn.investment
                    txn_idx += 1

            cum_units[i] = running_units
            cum_invested[i] = running_invested

        equity = cum_units * prices
        cost_basis = np.where(cum_units > 0, cum_invested / cum_units, 0)

        return pd.DataFrame({
            "price": prices,
            "cum_units": cum_units,
            "cum_invested": cum_invested,
            "equity": equity,
            "cost_basis": cost_basis,
            "unrealized_pnl": equity - cum_invested,
            "return_pct": np.where(
                cum_invested > 0,
                (equity - cum_invested) / cum_invested * 100,
                0,
            ),
        }, index=dates[:n])

    def _compare_vs_uniform(
        self,
        scores: np.ndarray,
        prices: np.ndarray,
        dates: pd.DatetimeIndex,
        guided_invested: float,
        guided_buys: int,
    ) -> Dict[str, float]:
        cfg = self.config
        n = len(prices)

        buy_indices = list(range(0, n, cfg.cadence_days))
        valid_buys = [
            i for i in buy_indices
            if not np.isnan(prices[i]) and prices[i] > 0
        ]

        if not valid_buys or guided_invested <= 0:
            return {"cost_improvement_pct": 0.0, "return_diff_pct": 0.0}

        uniform_per_buy = guided_invested / len(valid_buys)
        uniform_units = sum(
            uniform_per_buy / prices[i] for i in valid_buys
        )
        uniform_avg_cost = guided_invested / uniform_units if uniform_units > 0 else 0

        guided_avg_cost = guided_invested / sum(
            1 for _ in range(1)                          
        )
        final_price = prices[-1] if not np.isnan(prices[-1]) else prices[~np.isnan(prices)][-1]

        uniform_value = uniform_units * final_price
        uniform_return = (uniform_value - guided_invested) / guided_invested * 100 if guided_invested > 0 else 0

        cost_improvement = (uniform_avg_cost - uniform_avg_cost) / uniform_avg_cost * 100 if uniform_avg_cost > 0 else 0

        return {
            "uniform_avg_cost": float(uniform_avg_cost),
            "uniform_total_units": float(uniform_units),
            "uniform_final_value": float(uniform_value),
            "uniform_return_pct": float(uniform_return),
            "cost_improvement_pct": float(cost_improvement),
        }


def simulate_multi_asset_dca(
    asset_data: Dict[str, Dict[str, Any]],
    config: Optional[DCAPortfolioConfig] = None,
) -> Dict[str, PortfolioSimResult]:
    sim = DCAPortfolioSimulator(config)
    results = {}

    for symbol, data in asset_data.items():
        try:
            result = sim.run(
                scores=data["scores"],
                prices=data["prices"],
                dates=data["dates"],
                symbol=symbol,
            )
            results[symbol] = result
            logger.info(
                f"{symbol}: invested={result.total_invested:,.0f}, "
                f"return={result.total_return_pct:.2f}%, "
                f"tiers={result.tier_summary}"
            )
        except Exception as e:
            logger.error(f"DCA simulation failed for {symbol}: {e}")

    return results


def generate_dca_report(
    results: Dict[str, PortfolioSimResult],
    output_path: str = "backtest_logs/dca_portfolio_report.json",
) -> None:
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "assets": {},
    }

    for symbol, result in results.items():
        report["assets"][symbol] = {
            "total_invested": result.total_invested,
            "final_value": result.final_value,
            "total_return_pct": result.total_return_pct,
            "avg_cost_basis": result.avg_cost_basis,
            "total_fees": result.total_fees,
            "tier_summary": result.tier_summary,
            "vs_uniform": result.vs_uniform,
            "n_transactions": len(result.transactions),
            "config": result.config,
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"DCA report saved to {output_path}")
