# Trading Bot Data Leakage and Validation Audit Report

## 1. Executive Summary

The codebase contains **three critical** and **two high** data-leakage risks: (1) GMM regime in `indicators/hmm_regime.py` is fit on the full return series, so predictions at each time use a model that has seen the future; (2) HMM in `bot/regime.py` is fit on full returns and is unsafe for any live use; (3) `scripts/regime_cluster.py` uses KMeans on full-sample features, so regime labels and IC/deciles are in-sample; (4) `weights/ic_ewma.py` uses forward returns in `update()` and is high risk if used for live weighting; (5) `_simple_regime_detection` uses a global volatility threshold from the full sample. Validation and tuning are correctly implemented: walk-forward CV and purged K-fold with embargo are used in `validation/walkforward.py`, `validation/kfold.py`, and `validation/phase3_runner.py`; optimizers in `validation/tuning.py` and `validation/orchestrator.py` use out-of-sample metrics. Scripts that run single full backtests (e.g. noise_sensitivity, turnover_sensitivity) or full-sample statistics (regime_cluster, rolling_ic_weights) are high or critical risk if their outputs drive strategy or hyperparameters. Execution is realistic for the portfolio simulator (next-day-open execution, slippage in bps); validation layer adds impact and latency models but partial fills and liquidity caps are not implemented.

---

## 2. Leakage Findings Table (Phase 1)

| file | function | variable | method | uses_future_data? | reason | severity |
|------|----------|----------|--------|-------------------|--------|----------|
| indicators/hmm_regime.py | _gmm_regime_detection | gmm_ref | GaussianMixture.fit(full_returns) | Yes | GMM fit on entire return series; predict_proba at each i uses model that saw full sample | CRITICAL |
| bot/regime.py | fit_hmm | model | model.fit(returns) on full column | Yes | HMM fit on full returns; any live use of fitted model is leaked | CRITICAL |
| scripts/regime_cluster.py | main (clustering block) | X, labels | KMeans.fit_predict(X) on all valid bars | Yes | Clustering uses full-sample vol/trend; regime labels use future information | CRITICAL |
| weights/ic_ewma.py | update | y, raw_ic | np.mean(forward_returns[start+j : start+j+horizon]) | Yes | IC correlates signal with future returns; if update used at live t, horizon forward is future | HIGH |
| indicators/hmm_regime.py | _simple_regime_detection | threshold | np.percentile(valid_vol, vol_threshold_percentile) | Yes | Threshold computed over full valid_vol; applied at each i | HIGH |
| indicators/ldc.py | LDC.fit | _templates | fit(templates) bull/bear | Unclear | If templates built from same series later scored, leakage; call sites must be checked | MEDIUM |
| scripts/rolling_ic_weights.py | rolling IC | rolling_ic, last_ics | Rolling window IC vs forward returns | Unclear | Forward returns for evaluation; if weights fed to live scoring, indirect leakage | MEDIUM |
| indicators/normalization_refactor.py | _expanding_midrank_ecdf, canonical_normalize | pct_t, hist | Expanding ECDF with raw[:t+1] at each t | No | Only past and current at each index | SAFE |
| backend/indicators.py | calculate_z_score, SMA, RSI, MACD, etc. | rolling series | rolling(...).mean(), .std() | No | Pandas backward-looking rolling | SAFE |
| backend/backtest.py | _compute_rolling_scores | ind, scores | Per-bar loop using iloc[i] and rolling series up to i | No | Only past data at each bar | SAFE |
| backtester/portfolio_simulator.py | compute_scores_for_date_index | SMA, RSI, etc. | Backward-looking rolling then index by bar | No | Safe | SAFE |
| indicators/composite_refactor.py | composite + exit | exit_norm | canonical_normalize(exit_arr, ...) | No | Expanding ECDF in normalizer | SAFE |
| validation/objective.py | compute_gt_score | coeffs | np.polyfit(x, y, 1) | Depends | Call sites use equity curves from OOS backtest splits | SAFE when OOS |
| backend/scoring.py | calculate_statistical_deviation_score, etc. | z_scores, rules | Point-in-time indicator values | No | Uses precomputed indicators at current bar only | SAFE |
| indicators/volatility.py | realized_vol | vol_t | ret_window = log_returns[i-window:i], np.std(valid_rets) | No | Rolling window past-only | SAFE |
| indicators/volatility.py | volatility_percentile | pct_t | hist_vol = vol_series[:i] | No | Expanding past-only | SAFE |
| indicators/coupling.py | coupling_score | C_t | window_returns = all_returns[i-window:i], compute_shrinkage_covariance(valid_returns) | No | Rolling window past-only | SAFE |
| indicators/hawkes.py | _fit_hawkes_mle, estimate_hawkes | mu, alpha, beta | minimize on event_times, T_end | Depends | If event_times is rolling window up to t, safe; if full series, leakage | Depends on call site |
| validation/execution_model.py | apply_execution_costs | adv | volumes.rolling(20, min_periods=1).mean() | No | Backward-looking rolling | SAFE |
| bot/features.py | ofi_feature | series | compute_ofi with rolling window | No | Backward-looking | SAFE |
| backend/phase1_routes.py | Phase1 indicators | H_t, U_t, R_t, etc. | normalize_to_score, infer_regime_prob, volatility_regime_score, coupling_score | No (except regime) | Regime from hmm_regime is CRITICAL; others use rolling/expanding past-only | SAFE except regime |

---

## 3. Validation Findings (Phase 3)

| module | validation type | correct? | issue |
|--------|-----------------|----------|--------|
| validation/walkforward.py | Walk-forward CV (expanding or rolling train, test windows) | Yes | Proper time-ordered folds; used in phase3_runner and tests |
| validation/kfold.py | Purged K-fold with embargo | Yes | Purge + embargo to avoid leakage; used in tuning and phase3 |
| backtester/purged_validation.py | walk_forward_validate, PurgedKFold | Yes | Walk-forward and purged K-fold options |
| backend/backtest.py | Single backtest over one date range | N/A | Not walk-forward; one split. Acceptable for app backtest, not for optimization |
| validation/phase3_runner.py | Outer: walk-forward; inner: purged K-fold in tuning | Yes | Correct structure |
| validation/orchestrator.py | Splits from run_orchestrator; objective over splits | Yes | Backtest run per split; GT and Sharpe per split |
| Production DCA backtest (API) | Single run | N/A | No OOS; not used for parameter selection in API |

**Conclusion**: Validation and tuning use walk-forward or purged CV where it matters. The main backtest API is single-split by design; optimization must not use that single backtest metric.

---

## 4. Optimizer Findings (Phase 4)

| file | optimizer | objective metric | uses out-of-sample? | risk |
|------|-----------|------------------|---------------------|------|
| validation/tuning.py | Grid / random / Bayesian (skopt) | Sortino, IR, or IC via _evaluate_inner_cv | Yes | Inner purged K-fold; metrics OOS per fold. LOW |
| validation/mfbo.py | Optuna (multi-fidelity) | User objective_fn | Depends | Orchestrator passes GT mean over splits; OOS. LOW if splits true OOS |
| validation/orchestrator.py | MFBO / random trials | compute_gt_score (mean over splits), DSR on Sharpes | Yes | Backtest per split. LOW |
| scripts/noise_sensitivity.py | N/A (sweep) | total_return_pct | No | Single full backtest per run. HIGH RISK if used to choose params |
| scripts/turnover_sensitivity.py | N/A (sweep) | total_return_pct, trades | No | Single backtest per threshold. HIGH RISK if used to select threshold |
| scripts/suggest_threshold_from_deciles.py | Suggestion from deciles | — | Unclear | If suggestion used in production without OOS check, MEDIUM |
| scripts/rolling_ic_weights.py | N/A | Rolling IC to suggested weights | No | Forward returns for IC; if weights fed to production, HIGH RISK |
| scripts/regime_cluster.py | N/A | IC/deciles per regime | No | Regime labels from full-sample KMeans; CRITICAL leakage in regime definition |

**Summary**: Tuning and orchestrator use OOS (walk-forward + purged inner CV). Scripts that run single full backtests or full-sample statistics are HIGH/CRITICAL risk if their outputs drive strategy or hyperparameters without separate OOS validation.

---

## 5. Execution Model Findings (Phase 5)

| feature | implemented? | realistic? | notes |
|---------|--------------|------------|--------|
| Next-bar execution | Yes | Yes | backtester/portfolio_simulator.py: signal at T close, execute at T+1 open; pending_entries/pending_exits at next day open |
| Slippage model | Yes | Partial | execution_price(raw_price, side, slippage_bps): linear bps; no size-dependent impact in simulator. validation/execution_model has impact + noise for tuning |
| Latency | Yes (validation only) | Partial | simulate_latency in execution_model; not wired into portfolio_simulator event loop |
| Partial fills | No | No | All-or-nothing fills |
| Liquidity caps | No | No | No ADV/volume cap on order size |
| Spread simulation | Yes (validation) | Partial | execution_model: spread_bps, commission_bps; portfolio_simulator uses slippage_bps only |
| Same-close execution | Option | Unrealistic | SimConfig.execution can be same_close; default next_open is realistic |

---

## 6. Top 5 Critical Issues Ranked

1. **GMM regime (indicators/hmm_regime.py)** — Fit on full return series; predictions at each time use a model that has seen the future. CRITICAL.
2. **HMM fit (bot/regime.py)** — Fit on full series; any live use of the fitted model is leaked. CRITICAL.
3. **Regime clustering (scripts/regime_cluster.py)** — KMeans on full-sample features; regime labels and IC/deciles per regime are in-sample. CRITICAL.
4. **IC-EWMA weights (weights/ic_ewma.py)** — Uses forward returns in update(); if used at runtime for live weighting, future data is used. HIGH.
5. **Simple regime threshold (indicators/hmm_regime.py)** — _simple_regime_detection uses np.percentile(valid_vol, ...) over full valid_vol as threshold; global statistic applied at each t. HIGH.

---

## 7. Recommended Fix Order

1. **indicators/hmm_regime.py**: Replace full-series GMM fit with rolling/expanding window fit (fit on returns[i-window:i] at each i), or use only past data for threshold in simple regime.
2. **bot/regime.py**: Either fit HMM only on past data when generating R_t at each t (rolling/expanding HMM), or restrict fit_hmm to offline analysis and never use for live prediction.
3. **scripts/regime_cluster.py**: Do not use full-sample KMeans for regime assignment. Use rolling or expanding clustering, or treat script as descriptive only and do not use regime labels for strategy/tuning.
4. **weights/ic_ewma.py**: Ensure update() is only called with data up to current time and that forward returns used for IC are not available at decision time (e.g. use only for offline evaluation, not for live weight updates).
5. **tests/leakage_check.py**: Implement and run; fix any indicator that shows mismatch (full vs incremental) before further optimization or refactoring.
