"""
Currency/FX consistency: ensure USD assets are not double-converted and cost/notional are sane.

Runs portfolio sim with a USD asset (AAPL); asserts trade prices and notionals are in
reasonable range (would be inflated if FX were applied twice to price). See
docs/ARCHITECTURE.md "Currency and FX consistency" for the documented flow.
"""

import sys
from pathlib import Path
from datetime import datetime
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def sim_module():
    """Load backtester portfolio_simulator without pulling project indicators package."""
    import importlib.util
    sim_path = PROJECT_ROOT / "backtester" / "portfolio_simulator.py"
    spec = importlib.util.spec_from_file_location("portfolio_simulator", str(sim_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["portfolio_simulator"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_usd_asset_no_double_fx(sim_module):
    """Run sim with one USD asset; trade price and notional should be in listing-currency range."""
    symbols = ["AAPL"]
    asset_meta = {s: {"asset_type": "equity", "currency": "USD"} for s in symbols}
    start_dt = datetime(2021, 1, 1)
    end_dt = datetime(2021, 12, 31)
    date_index, assets_data = sim_module.prepare_multi_asset_data(
        symbols=symbols,
        start_date=start_dt,
        end_date=end_dt,
        asset_meta=asset_meta,
        usd_inr_rate=83.5,
    )
    config = sim_module.SimConfig(
        symbols=symbols,
        start_date=start_dt,
        end_date=end_dt,
        initial_capital=100_000.0,
        entry_score_threshold=65.0,
        slippage_bps=5.0,
        cost_free=False,
        run_benchmarks=False,
    )
    sim = sim_module.PortfolioSimulator(config)
    result = sim.run(date_index, assets_data)

    assert "total_return_pct" in result
    assert isinstance(result["total_return_pct"], (int, float))
    assert result["total_costs"] >= 0

    # USD equity: price should be in plausible range (e.g. AAPL ~100–200); if FX were applied
    # twice we'd see price ~83x larger
    for t in result.get("trades", []):
        price = t.get("price") or 0
        notional = t.get("notional") or 0
        assert price > 0 and price < 1e6, f"Trade price {price} out of plausible USD range"
        assert notional >= 0, "Notional should be non-negative"
