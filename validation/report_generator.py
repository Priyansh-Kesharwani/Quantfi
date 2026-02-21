"""
Phase B — HTML Report Generator.

Generates a self-contained HTML report for each validated asset, including:
  - Score distribution histograms
  - Walk-forward summary tables
  - Exit signal ROC curve (vs true regime flips)
  - λ(t) simulation plots
  - Entry_Score vs forward return heatmap
  - All metrics in tabular form

Output: /validation/reports/{symbol}_phaseB_report.html
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json
import logging

from validation.plots import (
    plot_score_distribution,
    plot_walkforward_summary,
    plot_hawkes_overlay,
    plot_score_return_heatmap,
    plot_exit_roc,
    plot_equity_curve,
)

logger = logging.getLogger(__name__)


_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phase B Validation Report — {symbol}</title>
<style>
    body {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: #0a0a0f;
        color: #e0e0e0;
        margin: 0; padding: 20px;
        line-height: 1.6;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{
        color: #ffd700;
        border-bottom: 2px solid #333;
        padding-bottom: 10px;
    }}
    h2 {{
        color: #90caf9;
        margin-top: 40px;
        border-left: 4px solid #ffd700;
        padding-left: 12px;
    }}
    .meta {{
        color: #888;
        font-size: 0.85em;
        margin-bottom: 30px;
    }}
    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 20px 0;
    }}
    .metric-card {{
        background: #1a1a2e;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #333;
    }}
    .metric-card .label {{
        color: #888;
        font-size: 0.8em;
        text-transform: uppercase;
    }}
    .metric-card .value {{
        color: #ffd700;
        font-size: 1.5em;
        font-weight: bold;
    }}
    .plot-section {{
        margin: 20px 0;
        text-align: center;
    }}
    .plot-section img {{
        max-width: 100%;
        border-radius: 8px;
        border: 1px solid #333;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }}
    th {{
        background: #1565c0;
        color: white;
        padding: 10px;
        text-align: center;
    }}
    td {{
        padding: 8px 10px;
        border-bottom: 1px solid #333;
        text-align: center;
    }}
    tr:hover {{ background: #1a1a2e; }}
    .pass {{ color: #4caf50; font-weight: bold; }}
    .fail {{ color: #f44336; font-weight: bold; }}
    .warn {{ color: #ff9800; }}
    .section-divider {{
        border: 0;
        height: 1px;
        background: linear-gradient(to right, transparent, #333, transparent);
        margin: 40px 0;
    }}
    .json-block {{
        background: #111;
        border-radius: 6px;
        padding: 12px;
        overflow-x: auto;
        font-family: monospace;
        font-size: 0.85em;
        white-space: pre-wrap;
        border: 1px solid #333;
    }}
</style>
</head>
<body>
<div class="container">
    <h1>📊 Phase B Validation Report — {symbol}</h1>
    <div class="meta">
        Generated: {timestamp} | Interval: {interval} | Bars: {n_bars}
    </div>

    {sections}

</div>
</body>
</html>
"""


def _metric_card(label: str, value: Any, fmt: str = ".3f") -> str:
    """Render a single metric card."""
    if isinstance(value, float):
        if np.isnan(value):
            v_str = "N/A"
        else:
            v_str = f"{value:{fmt}}"
    else:
        v_str = str(value)
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{v_str}</div>
    </div>
    """


def _img_tag(b64_png: str) -> str:
    """Embed a base64 PNG in an <img> tag."""
    return f'<img src="data:image/png;base64,{b64_png}" />'


def generate_report(
    symbol: str,
    entry_scores: pd.Series,
    exit_scores: pd.Series,
    prices: pd.Series,
    walkforward_result: Optional[Dict[str, Any]] = None,
    kfold_result: Optional[Dict[str, Any]] = None,
    hawkes_validations: Optional[List[Dict[str, Any]]] = None,
    hawkes_regimes: Optional[List[Dict[str, Any]]] = None,
    regime_flips: Optional[pd.Series] = None,
    interval: str = "1d",
    output_dir: str = "validation/reports",
    forward_horizon: int = 5,
) -> str:
    """Generate a full Phase B HTML report.

    Parameters
    ----------
    symbol : str
        Asset symbol.
    entry_scores, exit_scores : pd.Series
        Composite scores.
    prices : pd.Series
        Close price series.
    walkforward_result : dict or None
        WalkForwardResult.to_dict().
    kfold_result : dict or None
        KFoldResult.to_dict().
    hawkes_validations : list or None
        List of validation dicts from validate_estimation.
    hawkes_regimes : list or None
        List of regime simulation dicts from simulate_regime.
    regime_flips : pd.Series or None
        Binary labels for true regime flips.
    interval : str
        Data interval.
    output_dir : str
        Output directory for the HTML file.
    forward_horizon : int
        Forward return horizon for heatmap.

    Returns
    -------
    str
        Path to the generated HTML file.
    """
    sections = []

    # ── Section 1: Score Distribution ─────────────────────────
    sections.append("<h2>📊 Score Distributions</h2>")
    dist_b64 = plot_score_distribution(entry_scores, exit_scores, title=f"{symbol} — Score Distributions")
    sections.append(f'<div class="plot-section">{_img_tag(dist_b64)}</div>')

    # ── Section 2: Summary Metrics ────────────────────────────
    sections.append("<h2>📐 Summary Metrics</h2>")
    sections.append('<div class="metric-grid">')

    valid_entry = entry_scores.dropna()
    valid_exit = exit_scores.dropna()
    sections.append(_metric_card("Mean Entry", float(valid_entry.mean()) if len(valid_entry) > 0 else np.nan, ".1f"))
    sections.append(_metric_card("Mean Exit", float(valid_exit.mean()) if len(valid_exit) > 0 else np.nan, ".1f"))
    sections.append(_metric_card("Entry in [0,100]", "✅" if ((valid_entry >= 0) & (valid_entry <= 100)).all() else "❌"))
    sections.append(_metric_card("Exit in [0,100]", "✅" if ((valid_exit >= 0) & (valid_exit <= 100)).all() else "❌"))
    sections.append(_metric_card("Total Bars", len(prices)))
    sections.append(_metric_card("NaN Entry", int(entry_scores.isna().sum())))
    sections.append(_metric_card("NaN Exit", int(exit_scores.isna().sum())))

    sections.append("</div>")

    # ── Section 3: Walk-Forward CV ────────────────────────────
    if walkforward_result:
        sections.append("<hr class='section-divider'>")
        sections.append("<h2>📈 Walk-Forward Cross-Validation</h2>")
        wf_b64 = plot_walkforward_summary(walkforward_result)
        sections.append(f'<div class="plot-section">{_img_tag(wf_b64)}</div>')

        # Summary metrics
        summary = walkforward_result.get("summary", {})
        if summary:
            sections.append('<div class="metric-grid">')
            for k, v in summary.items():
                fmt = ".1%" if "rate" in k or "drawdown" in k else ".4f"
                if isinstance(v, int):
                    fmt = "d"
                sections.append(_metric_card(k.replace("_", " ").title(), v, fmt))
            sections.append("</div>")

    # ── Section 4: K-Fold CV ──────────────────────────────────
    if kfold_result:
        sections.append("<hr class='section-divider'>")
        sections.append("<h2>🧪 Purged K-Fold Cross-Validation</h2>")
        summary = kfold_result.get("summary", {})
        if summary:
            sections.append('<div class="metric-grid">')
            for k, v in summary.items():
                fmt = ".4f"
                if isinstance(v, int):
                    fmt = "d"
                sections.append(_metric_card(k.replace("_", " ").title(), v, fmt))
            sections.append("</div>")

        # Per-fold table
        folds = kfold_result.get("folds", [])
        if folds:
            sections.append("<table><tr><th>Fold</th><th>Test Period</th><th>Train</th><th>Test</th><th>Warnings</th></tr>")
            for f in folds:
                warns = ", ".join(f.get("warnings", [])) or "—"
                sections.append(
                    f"<tr><td>{f['fold_idx']}</td>"
                    f"<td>{f['test_start'][:10]}→{f['test_end'][:10]}</td>"
                    f"<td>{f['train_size']}</td>"
                    f"<td>{f['test_size']}</td>"
                    f"<td class='warn'>{warns}</td></tr>"
                )
            sections.append("</table>")

    # ── Section 5: Hawkes Simulations ─────────────────────────
    if hawkes_regimes or hawkes_validations:
        sections.append("<hr class='section-divider'>")
        sections.append("<h2>🧬 Hawkes λ(t) Simulation Tests</h2>")

        if hawkes_validations:
            sections.append("<table><tr><th>Regime</th><th>RMSE</th><th>Rel. RMSE</th><th>#Events</th><th>η</th><th>Pass</th></tr>")
            for v in hawkes_validations:
                status = '<span class="pass">✅</span>' if v["passed"] else '<span class="fail">❌</span>'
                sections.append(
                    f"<tr><td>{v['regime_name']}</td>"
                    f"<td>{v['rmse']:.4f}</td>"
                    f"<td>{v['relative_rmse']:.1%}</td>"
                    f"<td>{v['n_events']}</td>"
                    f"<td>{v['branching_ratio']:.3f}</td>"
                    f"<td>{status}</td></tr>"
                )
            sections.append("</table>")

        if hawkes_regimes:
            for reg in hawkes_regimes:
                b64 = plot_hawkes_overlay(
                    grid=reg["grid"],
                    true_intensity=reg["true_intensity"],
                    events=reg["events"],
                    regime_name=reg["regime_name"],
                )
                sections.append(f'<div class="plot-section">{_img_tag(b64)}</div>')

    # ── Section 6: Score vs Return Heatmap ────────────────────
    sections.append("<hr class='section-divider'>")
    sections.append("<h2>🔥 Entry Score vs Forward Return Heatmap</h2>")
    from validation.metrics import forward_returns as fwd_ret_fn
    fwd_ret = fwd_ret_fn(prices, horizon=forward_horizon)
    heatmap_b64 = plot_score_return_heatmap(
        entry_scores, fwd_ret,
        title=f"{symbol} — Entry Score vs {forward_horizon}d Forward Return"
    )
    sections.append(f'<div class="plot-section">{_img_tag(heatmap_b64)}</div>')

    # ── Section 7: Exit ROC ───────────────────────────────────
    if regime_flips is not None:
        sections.append("<hr class='section-divider'>")
        sections.append("<h2>📉 Exit Signal ROC Curve</h2>")
        roc_b64 = plot_exit_roc(exit_scores, regime_flips, title=f"{symbol} — Exit Score ROC")
        sections.append(f'<div class="plot-section">{_img_tag(roc_b64)}</div>')

    # ── Assemble HTML ─────────────────────────────────────────
    html = _HTML_TEMPLATE.format(
        symbol=symbol,
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        interval=interval,
        n_bars=len(prices),
        sections="\n".join(sections),
    )

    # Save
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_phaseB_report.html"
    out_path.write_text(html, encoding="utf-8")

    logger.info(f"Report saved to {out_path}")
    return str(out_path)
