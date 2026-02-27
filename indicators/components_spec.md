# Canonical indicator formulas and return contract

All refactor-path indicators must return the same shape and metadata so composite and validation can consume them uniformly.

---

## Standard return signature

Every indicator function returns:

```python
(pd.Series values, dict meta)
```

**meta** must include:

- `name`: str — indicator name (e.g. "OFI", "VWAP_Z")
- `window`: int — lookback/window used
- `n_obs`: int — number of valid observations
- `unit`: str — e.g. "dimensionless", "percent"
- `polarity`: str — "higher_favorable" or "lower_favorable"
- `warnings`: list — optional list of warning strings

---

## OFI (bar-level simplified)

- `sign_t = sign(p_t - p_{t-1})`
- `ofi_bar_t = sign_t * volume_t`
- `OFI_t = sum_{i=t-W+1}^{t} ofi_bar_i`

Window W from config `indicators.ofi.window`. Optional: normalize output with canonical normalizer.

---

## VWAP Z-score (displacement)

- `Z_vwap,t = (log p_t - log(VWAP_t)) / sigma_t^vol` (or price-based z-score in implementation)

VWAP_t = volume-weighted average price over a rolling window; sigma_t^vol from configurable vol window (e.g. `indicators.vwap_z.vol_window`). Implementation may use (p_t - VWAP_t)/sigma_t for stability.

---

## Hurst / Persistence H_t

- Implement DFA or wavelet estimator; document estimator bias.
- Fallback: R/S (rescaled range) with unit tests on synthetic FBM.
- Config: `indicators.hurst.method`, `indicators.hurst.window`.

---

## Hawkes intensity (exponential kernel)

- `λ(t) = μ + sum_{t_i < t} α exp(-β(t - t_i))`

Implement via `tick` library if available; else vectorized or numba incremental. Return (intensity series, meta).

---

## LDC (Lorentzian Distance Classifier)

- Lorentzian distance: `d_L(x,y) = sum_i log(1 + ((x_i - y_i)/γ_i)^2)`
- Similarity: `s(x) = 1 / (1 + exp(κ * (d_bull(x) - d_bear(x))))`

Templates (bull/bear) from normalized component windows; kappa and gamma from config.
