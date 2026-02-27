# Refactor-First Plan — Mathematics and Validation

Single locked reference for the research-grade composite Entry/Exit formulation, normalization, indicators, weighting, and validation protocol. Only after acceptance (Phase R3) proceed to trading-bot design.

---

## Summary of objectives

- Produce a single, research-grade, dimensionless, regime-robust composite Entry_Score / Exit_Score formulation.
- Make the formula interpretable, config-driven, and testable (deterministic runs, audit logs).
- Validate with unit tests, synthetic-process tests (FBM/OU/Hawkes), walk-forward + purged-K CV, ablation, cost sensitivity and reproducibility.
- Only after acceptance, design and implement trading-bot integration.

---

## Phase R1 — Mathematical refactor (formulas)

### 1. Canonical normalization (single source of truth)

Expanding-ECDF with midrank tie rule, then inverse-normal, then sigmoid.

- `p_t = (rank_less_t + 0.5 * rank_equal_t) / n_t`
- `z_t = Phi^{-1}(clip(p_t, eps, 1-eps))`
- `s_t = 1 / (1 + exp(-k * z_t))`
- Polarity: if larger raw is harmful, use `s_t <- 1 - s_t`.
- Config: `k`, `eps`, `mode in {exact, approx}`.

### 2. Canonical indicator formulas

- **OFI (bar-level):** `sign_t = sign(p_t - p_{t-1})`, `ofi_bar_t = sign_t * volume_t`, `OFI_t = sum_{i=t-W+1}^{t} ofi_bar_i`.
- **VWAP Z-score:** `Z_vwap,t = (log p_t - log(VWAP_t)) / sigma_t^vol`.
- **Hurst / Persistence H_t:** DFA or wavelet; fallback R/S; unit tests on synthetic FBM.
- **Hawkes intensity:** `λ(t) = μ + sum_{t_i < t} α exp(-β(t - t_i))`.
- **LDC:** `d_L(x,y) = sum_i log(1 + ((x_i - y_i)/γ_i)^2)`; `s(x) = 1/(1 + exp(κ(d_bull(x) - d_bear(x))))`.

Each indicator returns `(pd.Series values, dict meta)` with `meta: {name, window, n_obs, unit, polarity, warnings}`.

### 3. Composite structure (symbolic)

- **Step A — persistence gating:** `g_pers(H_t) = sigmoid(k_pers * (H_t - 0.5)) in (0,1)`.
- **Step B:** All indicator outputs normalized with canonical normalizer; polarity aligned (higher = more favorable to buy).
- **Step C:** `Opp_t = trimmed_mean(T_t, U_t * g_pers(H_t), LDC_t, O_t)`; trim_frac configurable.
- **Step D:** `Gate_t = C_t * L_t * 1(R_t >= r_thresh)`.
- **Step E:** `RawFavor_t = Opp_t * Gate_t`; `CompositeScore_t = 100 * clip(0.5 + (RawFavor_t - 0.5) * S_scale, 0, 1)`.
- **Exit:** `ExitRaw_t = γ1*TBL_t + γ2*OFI_rev_t + γ3*lambda_decay_t`; normalize with canonical normalizer → ExitScore_t in [0,100].

All `k_*`, `γ_*`, `S_scale`, `r_thresh` in `config/phase_refactor.yml` — no hard-coded numbers in code.

### 4. Weighting (IC-EWMA)

- `w_tilde_i(t) = exp(α * (EWMA_IC_i(t) - μ_IC(t)) / (σ_IC(t) + ε))`
- Normalize to simplex with shrinkage λ; clip per-step change `|w_i(t) - w_i(t-1)| <= δ_max`.

---

## Phase R2 — Testing and validation plan

- Unit tests: normalizer determinism, OFI/VWAPz/Hurst/Hawkes/LDC on synthetic data; determinism (two runs → same SHA256).
- Component IC and decile tests at horizons [5, 20, 60, 252]; pass condition: IC > 0.03 for at least one horizon and upper deciles statistically higher, or documented rationale.
- Backtest harness: walk-forward outer (e.g. train 3y / test 1y), inner purged K-fold; tuning only on train; OOS Sortino, CAGR, maxDD, turnover, IC stability.
- Ablation (leave-one-out), parameter sweeps, cost sensitivity (zero / conservative / realistic).
- Hawkes stress: synthetic scenarios; λ RMSE within tolerance.
- Artifacts under `validation/artifacts/{runid}/`: scores.parquet, breakdown.parquet, trades.csv, metrics.json, tuning_trace.json; determinism: same runid_seed → identical metrics.json (SHA256).

---

## Phase R3 — Acceptance gating (stop / go)

Minimum gating criteria (all must be true):

1. CompositeScore code, canonical normalizer, and component functions implemented and unit-tested.
2. Determinism check passes (e.g. 3 identical runs with same runid/seed).
3. Walk-forward: composite strategy shows positive realistic net-of-cost returns for at least 3 assets OR meaningful risk reduction vs baseline and stable IC for at least one horizon.
4. Execution model validated (Hawkes + slippage stress) within allowed tolerances.
5. No look-ahead or currency double-conversion bugs.

**Only after acceptance, proceed to Phase R4 and trading-bot design.** Run `python scripts/refactor_gating.py --runid <runid>` to evaluate the five gating criteria; exit 0 only if all pass.

---

## Phase R4 — Deliverables

- **Refactored code:** `indicators/normalization_refactor.py` (canonical ECDF→z→sigmoid), `indicators/refactor_components.py` (OFI, VWAP Z, Hurst, Hawkes, LDC with (Series, meta)), `indicators/composite_refactor.py` (Opp, Gate, RawFavor, CompositeScore, Exit from `config/phase_refactor.yml`), `indicators/components_spec.md`, `weights/` (IC-EWMA + Kalman stub).
- **Validation:** `validation/run_ic_decile.py`, `validation/refactor_runner.py`, `scripts/verify_refactor_determinism.py`, `scripts/hawkes_stress_refactor.py`, `scripts/ablation_refactor.py`, `scripts/refactor_gating.py`, `scripts/run_refactor_checks.sh`.
- **Config:** `config/phase_refactor.yml` (determinism, normalization, indicators, weighting, composite, exit, testing).
- **Tests:** `tests/test_normalization_refactor.py`, `tests/test_component_formulas.py`, `tests/test_composite_refactor.py`, `tests/test_weights.py`.
- **Artifacts:** `validation/artifacts/{runid}/` with `metrics.json`, `scores.parquet`, `breakdown.parquet` (after refactor_runner / verify_refactor_determinism).
- This spec (`REFRACTOR_MATHEMATICS.md`) is the locked human-readable reference for frontend/backend when integrating the trading bot.
