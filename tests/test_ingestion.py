"""
Unit tests for bot.ingestion.build_bars.

Deterministic behavior on fixture tests/fixtures/tick_sample.csv.
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.bot.ingestion import build_bars

FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "tick_sample.csv"

def test_build_bars_import():
    """Module and function are importable."""
    assert build_bars is not None

def test_build_bars_requires_timestamp():
    """Raises if no timestamp column and no datetime index."""
    df = pd.DataFrame({"price": [1.0, 2.0], "size": [1, 1]})
    with pytest.raises(ValueError, match="timestamp"):
        build_bars(df, "1min")

def test_build_bars_requires_price():
    """Raises if no price-like column."""
    df = pd.DataFrame({"timestamp_utc": pd.date_range("2020-01-01", periods=3, freq="s", tz="UTC"), "qty": [1, 1, 1]})
    with pytest.raises(ValueError, match="price"):
        build_bars(df, "1min")

@pytest.mark.parametrize("freq", ["1min", "2min"])
def test_build_bars_fixture_shape_and_columns(freq):
    """On tick_sample.csv, output has UTC index and required columns."""
    if not FIXTURE_PATH.exists():
        pytest.skip("fixture tests/fixtures/tick_sample.csv not found")
    ticks = pd.read_csv(FIXTURE_PATH)
    bars = build_bars(ticks, freq)
    required = ["open", "high", "low", "close", "volume", "bid", "ask", "vwap"]
    for c in required:
        assert c in bars.columns, f"missing column {c}"
    assert bars.index.tz is not None
    assert str(bars.index.tz) == "UTC"
    assert len(bars) >= 1
    assert bars["volume"].min() >= 0

def test_build_bars_deterministic():
    """Same ticks and freq produce identical output (hash of values)."""
    if not FIXTURE_PATH.exists():
        pytest.skip("fixture tests/fixtures/tick_sample.csv not found")
    ticks = pd.read_csv(FIXTURE_PATH)
    b1 = build_bars(ticks, "1min")
    b2 = build_bars(ticks, "1min")
    h1 = hashlib.sha256(b1.values.tobytes()).hexdigest()
    h2 = hashlib.sha256(b2.values.tobytes()).hexdigest()
    assert h1 == h2, "Two runs must produce identical bars"

def test_build_bars_1min_fixture_values():
    """On tick_sample.csv with 1min freq, first bar OHLC and volume are consistent."""
    if not FIXTURE_PATH.exists():
        pytest.skip("fixture tests/fixtures/tick_sample.csv not found")
    ticks = pd.read_csv(FIXTURE_PATH)
    bars = build_bars(ticks, "1min")
    first = bars.iloc[0]
    assert first["open"] == 100.0
    assert first["high"] == 100.5
    assert first["low"] == 99.8
    assert first["close"] == 100.2
    assert first["volume"] == 500.0
    assert abs(first["vwap"] - 100.03) < 0.01
