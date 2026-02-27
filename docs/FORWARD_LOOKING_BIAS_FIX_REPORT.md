# Forward-Looking Bias Fix — Implementation Report

## 1. Files Modified

| File | Change type |
|------|-------------|
| `indicators/hmm_regime.py` | Modified |
| `bot/regime.py` | Modified |
| `weights/ic_ewma.py` | Modified |
| `scripts/regime_cluster.py` | Header only |
| `scripts/rolling_ic_weights.py` | Header only |
| `scripts/noise_sensitivity.py` | Header only |
| `scripts/turnover_sensitivity.py` | Header only |
| `utils/__init__.py` | New |
| `utils/leakage_guard.py` | New |
| `bot/__init__.py` | Export added |
| `tests/leakage_check.py` | Comparison fix (inf handling) |
| `tests/test_bot_regime.py` | Test for `regime_probability_rolling` |

---

## 2. What Was Changed

**indicators/hmm_regime.py**
- **GMM (_gmm_regime_detection):** Removed single fit on full return series. For each index `i`, fit GaussianMixture on `returns[i-window:i]` only; predict regime at `i` from that model. Added `min_samples` (default `max(window//2, n_states*2)`); if valid past samples < `min_samples`, set `prob_stable[i] = FALLBACK_REGIME` (0.5). Window remains configurable via `HMMRegimeConfig.window`; `min_samples` configurable via `HMMRegimeConfig.min_samples`.
- **Simple regime (_simple_regime_detection):** Threshold no longer uses full `valid_vol`. For each `i`, volatility history is `vol[window:i]`; `threshold_i = np.percentile(valid_hist, vol_threshold_percentile)`. Only past-and-current vol used at each `i`. Fallback regime 0.5 when insufficient history.

**bot/regime.py**
- **Rolling HMM:** Added `regime_probability_rolling(returns_df, n_states, window, refit_every, ...)`. For each time index `i`, fits HMM on past only: `returns[max(0,i-window):i]` (or `returns[:i]` if `window` is None). Prediction at `i` uses that model only. Refit every `refit_every` bars (default 63) to limit cost. `_fit_hmm_on_past(returns_1d, end_idx, n_states, window, ...)` performs the past-only fit.
- **fit_hmm / predict_state_prob:** Left unchanged for backward compatibility. No change to existing call sites; new call sites that need leak-free regime should use `regime_probability_rolling`.

**weights/ic_ewma.py**
- **Mode flag:** Constructor accepts `mode="offline"` (default) or `mode="live"`. In `update()`, if `mode == "live"` then `RuntimeError(LIVE_FORWARD_MSG)` is raised immediately (forward returns are never used in live mode). Config can set `"mode": "live"` or `"mode": "offline"`.

**scripts (4 files)**
- First line after shebang: `# RESEARCH ONLY — DO NOT USE OUTPUT FOR STRATEGY PARAMETERS`

**utils/leakage_guard.py**
- `assert_past_only(series, index)`: Ensures `0 <= index < len(series)`; raises `ValueError` with `LEAKAGE_MSG_BOUNDS` otherwise. Supports numpy array and pandas Series.
- `assert_slice_past_only(current_index, start, end)`: Ensures `end <= current_index + 1` and `0 <= start <= end`; raises if slice would use future data.

**tests/leakage_check.py**
- In `_compare_at_t`, when both values are infinite, match is based on same sign (so `-inf` vs `-inf` is a match). This removes false mismatches from `nan` when comparing two infinities and yields 0 mismatches for the backend indicator set.

---

## 3. Proof No Future Data Used

- **GMM (hmm_regime):** At index `i`, fit uses `returns[i-window:i]` and prediction uses that model on `valid_returns[-1:]` (the value at `i`). No access to `returns[i+1:]`.
- **Simple regime (hmm_regime):** At index `i`, `hist_vol = vol[window:i]` and `threshold_i = np.percentile(valid_hist, ...)`. Only `vol` up to `i` used.
- **HMM (bot/regime):** `regime_probability_rolling` fits on `returns[max(0,i-window):i]` (or `returns[:i]`) and predicts at `i`. No data after `i` used.
- **IC_EWMA (weights/ic_ewma):** In `mode="live"`, `update()` raises before using `forward_returns`; in `mode="offline"`, call sites are responsible for using only in evaluation contexts.

---

## 4. Edge Cases Handled

- **GMM:** `n < window` → return all NaN. Per `i`: `len(valid) < min_samples` → `prob_stable[i] = FALLBACK_REGIME`. GMM fit failure at `i` → `prob_stable[i] = FALLBACK_REGIME`, no exception.
- **Simple regime:** `n` or `window` 0 → safe window; per `i` fewer than `window//2` valid history → skip (leave `prob_stable[i]` at fallback 0.5).
- **HMM rolling:** Insufficient past for fit at `i` → `_fit_hmm_on_past` returns `None` → row `i` left NaN. `predict_proba` failure → row left as-is (NaN).
- **IC_EWMA:** Invalid `mode` in constructor → `ValueError`. `mode="live"` and `update()` called → `RuntimeError`.
- **leakage_guard:** `index` out of range or non-integer → `ValueError`/`TypeError`. `assert_slice_past_only(current_index, start, end)` with `end > current_index+1` → `ValueError`.
- **leakage_check:** Both values ±inf with same sign → treated as match so mismatch count is 0 for the current indicator suite.

---

## 5. Remaining Risks

- **fit_hmm / predict_state_prob:** Still fit on full series when used as today. Any live or backtest path that calls `fit_hmm(full_returns)` then `predict_state_prob(df, model)` still uses a model trained on the full sample. Mitigation: use `regime_probability_rolling` for any leak-free regime series; reserve `fit_hmm` for offline analysis only.
- **IC_EWMA:** Default remains `mode="offline"`. Call sites that feed weights into live or backtest must set `mode="live"` and not call `update()` with forward returns, or use a different weighting path that does not rely on forward IC.
- **LDC / rolling_ic_weights / regime_cluster:** Not changed. Scripts are marked RESEARCH ONLY; outputs must not drive strategy parameters without separate OOS validation.
- **leakage_guard:** Only enforces valid index and slice bounds; it does not inspect whether callers use full-series statistics elsewhere. Callers must use `assert_past_only` / `assert_slice_past_only` at decision points and ensure all stats are computed from past-only slices.
