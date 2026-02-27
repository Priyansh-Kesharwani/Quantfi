# ECDF / Normalization Past-Only Audit

All ECDF and normalization calls use only past (and optionally current) data at each timestamp. No global statistics over the full series are used for scoring.

| Location | Function | Data used at time t | Past-only? |
|----------|----------|---------------------|------------|
| indicators/normalization.py | expanding_percentile | historical = series[:t]; pct = proportion of valid_hist <= current_value | Yes (strict past) |
| indicators/normalization.py | normalize_to_score | Calls expanding_percentile | Yes |
| indicators/normalization.py | expanding_ecdf_sigmoid | Calls normalize_to_score | Yes |
| indicators/normalization_refactor.py | _expanding_midrank_ecdf | hist = raw[:t+1]; rank over valid observations in [0,t] | Yes (current + past) |
| indicators/normalization_refactor.py | canonical_normalize | Calls _expanding_midrank_ecdf or expanding_percentile | Yes |
| indicators/composite_refactor.py | exit_norm | canonical_normalize(exit_arr, ...) | Yes |
| indicators/volatility.py | volatility_percentile | hist_vol = vol_series[:i] | Yes |
| indicators/ofi.py | OFI normalize | expanding_ecdf_sigmoid | Yes |
| indicators/hawkes.py | intensity normalize | expanding_ecdf_sigmoid | Yes |

No `.shift()` is required for these implementations: the expanding window is explicit (series[:t] or raw[:t+1]). No full-sample percentile, mean, or std is used for the value at t.
