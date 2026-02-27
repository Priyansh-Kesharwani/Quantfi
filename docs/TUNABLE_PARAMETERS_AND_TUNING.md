# Tunable Parameters and How They Are Tuned

## 1. Phase 1 config (indicator / scoring) — 5 surfaced tunables

**File:** `config/phase1.yml`  
**Section:** top-level `tunable`

Only these five parameters are designated as tunable; all others in the file are locked defaults.

| Parameter | Default | Meaning |
|-----------|---------|--------|
| `normalization_min_obs` | 100 | Min observations before first valid ECDF/normalization output |
| `normalization_sigmoid_k` | 1.0 | Sigmoid steepness in ECDF → score pipeline |
| `windows_hmm_regime` | 252 | Regime (HMM/GMM) lookback in bars |
| `indicator_min_history_rows` | 200 | Min rows before backend indicators return values |
| `backtest_buy_dip_threshold_default` | 60 | Default entry score threshold for buy-dip backtest |

**How they are tuned:**  
There is no single automated script that optimizes these five. They are the only phase1 knobs intended to be changed when you run optimization or sweeps. To tune them you would:

- Manually edit `config/phase1.yml` under `tunable`, or  
- Use Phase 3 tuning (see below) if your pipeline maps Phase 3 `search_space` keys (e.g. `lookback_*`, `S_scale`) into phase1-style config, or  
- Run your own grid/sweep over the five and plug best values back into `tunable`.

The rest of `phase1.yml` (hmm, normalization, windows, backend rules, etc.) is **locked**; change only the `tunable` block for optimization.

---

## 2. CPCV + MFBO (portfolio simulator) — strategy/execution params

**Entry point:** `python scripts/tune.py --config config/<tuning_cpcv*.yml>`  
**Config files:** `config/tuning_cpcv.yml`, `config/tuning_cpcv_tuned.yml`, `config/tuning_cpcv_bot.yml`, `config/tuning_cpcv_debug.yml`  
**Logic:** `validation/orchestrator.run_orchestrator` → `validation/mfbo.run_mfbo` (Optuna TPE)

### 2.1 Search space (YAML `mfbo.search_space`)

**Common (portfolio simulator):**

| Parameter | Example range | Role |
|-----------|----------------|------|
| `entry_score_threshold` | [58, 62, …, 78] or [60, 65, 70, 75, 80] | Enter when composite score > this |
| `max_positions` | [4, 6, 8, 10, 12, 15] or [5, 10, 15] | Cap on open positions |
| `slippage_bps` | [3, 5, 8, 12] | Fixed base slippage (bps) |
| `atr_trail_mult` | [2.0, 2.5, 3.0] | Trailing stop = peak − mult × ATR |
| `score_rel_mult` | [0.35, 0.4, 0.5] | Exit when score < entry_score × this |
| `score_abs_floor` | [30, 35, 40] | Exit when score < this level |
| `max_holding_days` | [20, 25, 30] | Max holding period (calendar days) |

**Bot-only** (`tuning_cpcv_bot.yml`, when `use_bot: true`):

| Parameter | Example range | Role |
|-----------|----------------|------|
| `kappa_tp` | [1.5, 2.0, 2.5, 3.0] | Take-profit scaling |
| `kappa_sl` | [1.0, 1.5, 2.0] | Stop-loss scaling |
| `T_max` | [30, 50, 70] | Max holding (bot) |

### 2.2 How tuning runs

1. **Data:** Load multi-asset data for `data.symbols` and `data.start_date` / `end_date`.
2. **Splits:** CPCV (Combinatorial Purged Cross-Validation) from `validation.validator`: `cpcv.n_folds`, `k_test`, `horizon_bars`, `embargo_bars`.
3. **Objective:** For each trial config, run backtest on each **test** fold; compute **GT-Score** per fold; objective = **mean GT-Score** across folds (maximized).
4. **Optimizer:** Optuna TPE (`run_mfbo`). If Optuna is missing, falls back to random sampling over `search_space` for `n_trials`.
5. **Selection:** After all trials, **Deflated Sharpe Ratio (DSR)** is computed per config (across folds). If any config has DSR ≥ `dsr_min`, the one with highest mean GT-Score among those is written to `winning_config.json`; otherwise the best mean GT-Score config is still written (with `dsr_passed: false`).

So: **tuned by** mean GT-Score (OOS per fold), with DSR as a significance gate; **optimizer** = Optuna TPE over the YAML `search_space`.

---

## 3. Phase 3 nested CV (signal/strategy layer) — Sortino / IR / IC

**Entry point:** Phase 3 runner (e.g. `validation/phase3_runner.py`) using `config/phase3.yml`.  
**Logic:** `validation/tuning.run_tuning` with inner purged K-fold; outer loop can be walk-forward (see `phase3.yml` and walkforward settings).

### 3.1 Tuning config (`config/phase3.yml` → `tuning`)

| Config key | Default | Role |
|------------|---------|------|
| `method` | `"random_search"` | `grid` \| `random_search` \| `bayesian` |
| `n_trials` | 50 | Number of trials (or grid size) |
| `objective` | `"sortino"` | `sortino` \| `ir` \| `ic` (per-fold metric) |
| `lambda_var` | 0.5 | Penalty on variance of fold metrics; score = median(metric) − λ×std(metric) |
| `inner_n_splits` | 5 | Inner purged K-fold splits |
| `inner_embargo` | 20 | Embargo bars in inner CV |
| `search_space` | (see below) | Parameter grid / search space |

### 3.2 Search space (phase3.yml `tuning.search_space`)

| Parameter | Example values | Role |
|-----------|----------------|------|
| `lookback_short` | [10, 20, 40] | Short lookback |
| `lookback_medium` | [40, 80, 120] | Medium lookback |
| `lookback_long` | [120, 200, 252] | Long lookback |
| `k_pers` | [1.0, 3.0, …, 12.0] | Persistence scaling |
| `S_scale` | [0.5, 0.8, 1.0, …, 3.0] | Score scaling (entry) |
| `S_scale_exit` | [0.5, …, 3.0] | Score scaling (exit) |
| `ofi_window` | [5, 10, 20, 40, 60] | OFI window |
| `ldc_kappa` | [0.5, 1.0, 2.0, 3.0] | LDC kappa |
| `ldc_feature_window` | [3, 5, 8, 16, 32] | LDC feature window |
| `hawkes_decay` | [0.5, 1.0, 2.0, 5.0] | Hawkes decay |

### 3.3 How tuning runs

1. **Inner CV:** Purged K-fold over training segment; each fold yields one metric (Sortino, IR, or IC).
2. **Trial score:** S(θ) = median(fold_metrics) − λ_var × std(fold_metrics).
3. **Search:**  
   - **grid:** full Cartesian product of `search_space`.  
   - **random_search:** `n_trials` random draws from `search_space`.  
   - **bayesian:** scikit-optimize (e.g. gp_minimize) over `search_space`, `n_calls = n_trials`.
4. **Best:** Config with highest S(θ) and its trial history are returned / persisted.

So: **tuned by** median Sortino (or IR/IC) minus variance penalty over inner OOS folds; **optimizer** = grid / random / Bayesian depending on `method`.

---

## 4. Signal sweep (DCA thresholds / cadence) — OOS Sortino ranking

**Module:** `backtester/signal_sweep.py`  
**Not an optimizer:** Discrete sweep over thresholds and multipliers; results are **ranked** by an OOS metric.

### 4.1 Parameters swept

| Parameter | Default / usage | Role |
|-----------|------------------|------|
| `thresholds` | e.g. 40–80 step 5 | Score threshold above which DCA invests more |
| `multipliers` | e.g. [1.0, 1.25, 1.5, 2.0] | Investment multiplier when score ≥ threshold |
| `frequency` | 5 | Invest every N days |
| `oos_start_idx` | optional | If set, only data from this index onward is used (OOS segment) |

### 4.2 How “tuning” works

1. **Sweep:** `sweep_dca_thresholds(scores, prices, config)` (and optionally `sweep_cadence_and_threshold`) builds a DCA equity curve per (threshold, multiplier) over the relevant segment (full or OOS).
2. **Metrics:** Each row gets `return_pct` and **`sortino_oos`** (Sortino of period returns on that segment).
3. **Ranking:** `rank_sweep_results(..., primary_metric="sortino_oos", ...)` sorts by OOS Sortino (default); `compute_sweep_heatmap(..., metric="sortino_oos")` uses the same metric.

So: **“tuning”** = choose the (threshold, multiplier, cadence) with best **OOS Sortino** from the sweep table (no continuous optimizer).

---

## 5. Summary table

| Where | What is tuned | Objective / metric | Optimizer / method |
|-------|----------------|--------------------|--------------------|
| **phase1.yml `tunable`** | 5 params (normalization, regime window, indicator history, buy-dip threshold) | Not automated; you change by hand or wire to your own optimizer | Manual or custom |
| **scripts/tune.py** (CPCV+MFBO) | entry_score_threshold, max_positions, slippage_bps, exit params (and bot params if use_bot) | Mean GT-Score over CPCV test folds; DSR ≥ dsr_min preferred | Optuna TPE (or random if no Optuna) |
| **Phase 3** (phase3_runner + tuning.py) | lookbacks, S_scale, OFI/LDC/Hawkes params in phase3.yml | S(θ) = median(Sortino/IR/IC) − λ×std over inner purged folds | grid / random_search / bayesian |
| **signal_sweep** | thresholds, multipliers, cadence | OOS Sortino (and return_pct) | Sweep + rank by sortino_oos |

---

## 6. Quick commands

- **CPCV + MFBO (portfolio sim):**  
  `python scripts/tune.py --config config/tuning_cpcv_tuned.yml [--output-dir DIR] [--seed N]`
- **Phase 3 (nested CV, phase3.yml):**  
  Run via `validation/phase3_runner` with config pointing at `phase3.yml` (script entry point may be in `scripts/` or docs).
- **DCA sweep (rank by OOS Sortino):**  
  Use `backtester.signal_sweep.sweep_dca_thresholds` with `SweepConfig(oos_start_idx=...)` then `rank_sweep_results(df, primary_metric="sortino_oos")`.
