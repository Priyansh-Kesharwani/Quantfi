"""Backend API routes for the crypto trading bot."""

import asyncio
import logging
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from crypto.calendar import TIMEFRAME_TO_MS, bars_per_day

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/crypto")

_TF_DEFAULT_RANGE = {
    "5m": timedelta(weeks=2),
    "15m": timedelta(days=30),
    "1h": timedelta(days=90),
    "4h": timedelta(days=180),
    "1d": timedelta(days=730),
}

MIN_BARS = 200
MAX_BARS = 10_000

def _compute_n_bars(start: datetime, end: datetime, tf: str) -> int:
    delta_ms = (end - start).total_seconds() * 1000
    tf_ms = TIMEFRAME_TO_MS.get(tf, 3_600_000)
    return max(1, int(delta_ms / tf_ms))

_market_cache: Dict[str, Any] = {"markets": [], "ts": 0.0, "ttl": 600}
_OFFLINE_FALLBACK = [
    {"symbol": "BTC/USDT:USDT", "base": "BTC", "name": "Bitcoin", "price": 42000, "volume_24h": 30e9},
    {"symbol": "ETH/USDT:USDT", "base": "ETH", "name": "Ethereum", "price": 2500, "volume_24h": 12e9},
    {"symbol": "SOL/USDT:USDT", "base": "SOL", "name": "Solana", "price": 100, "volume_24h": 3e9},
    {"symbol": "BNB/USDT:USDT", "base": "BNB", "name": "BNB", "price": 320, "volume_24h": 1.5e9},
    {"symbol": "XRP/USDT:USDT", "base": "XRP", "name": "XRP", "price": 0.55, "volume_24h": 1e9},
    {"symbol": "DOGE/USDT:USDT", "base": "DOGE", "name": "Dogecoin", "price": 0.08, "volume_24h": 800e6},
    {"symbol": "ADA/USDT:USDT", "base": "ADA", "name": "Cardano", "price": 0.35, "volume_24h": 500e6},
    {"symbol": "AVAX/USDT:USDT", "base": "AVAX", "name": "Avalanche", "price": 28, "volume_24h": 400e6},
    {"symbol": "LINK/USDT:USDT", "base": "LINK", "name": "Chainlink", "price": 14, "volume_24h": 350e6},
    {"symbol": "DOT/USDT:USDT", "base": "DOT", "name": "Polkadot", "price": 5.5, "volume_24h": 200e6},
]

def _load_markets_sync(exchange_id: str = "binance") -> List[Dict[str, Any]]:
    """Load all USDT linear perpetuals from exchange with live prices (blocking)."""
    try:
        import ccxt
        cls = getattr(ccxt, exchange_id)
        ex = cls({"options": {"defaultType": "swap"}, "enableRateLimit": True})
        ex.load_markets()

        perp_symbols = []
        for sym, info in ex.markets.items():
            if info.get("linear") and info.get("quote") == "USDT" and info.get("swap"):
                perp_symbols.append((sym, info))

        tickers = {}
        try:
            tickers = ex.fetch_tickers([s for s, _ in perp_symbols[:200]])
        except Exception:
            try:
                tickers = ex.fetch_tickers()
            except Exception as te:
                logger.warning("fetch_tickers failed: %s", te)

        results = []
        for sym, info in perp_symbols:
            t = tickers.get(sym, {})
            last_price = float(t.get("last", 0) or 0)
            vol_24h = float(t.get("quoteVolume", 0) or 0)
            results.append({
                "symbol": sym,
                "base": info.get("base", ""),
                "name": info.get("base", sym),
                "price": last_price,
                "volume_24h": vol_24h,
            })

        results.sort(key=lambda x: x["volume_24h"], reverse=True)

        for i, m in enumerate(results):
            if i < 10:
                m["category"] = "Major"
            elif i < 30:
                m["category"] = "Mid Cap"
            elif i < 80:
                m["category"] = "Alt"
            else:
                m["category"] = "Micro"

        return results
    except Exception as e:
        logger.warning("Failed to load markets from %s: %s", exchange_id, e)
        return []

async def _get_markets(exchange_id: str = "binance") -> List[Dict[str, Any]]:
    """Return cached market catalog, refreshing if TTL expired."""
    now = _time.time()
    if _market_cache["markets"] and (now - _market_cache["ts"]) < _market_cache["ttl"]:
        return _market_cache["markets"]

    markets = await asyncio.to_thread(_load_markets_sync, exchange_id)
    if markets:
        _market_cache["markets"] = markets
        _market_cache["ts"] = now
        return markets

    if _market_cache["markets"]:
        return _market_cache["markets"]

    for m in _OFFLINE_FALLBACK:
        m.setdefault("category", "Major")
    return _OFFLINE_FALLBACK

class CryptoBacktestRequest(BaseModel):
    symbol: str = "BTC/USDT:USDT"
    timeframe: str = "1h"
    initial_capital: float = 10_000.0
    strategy_mode: str = "adaptive"
    leverage: float = 3.0
    cost_preset: str = "BINANCE_FUTURES_TAKER"
    entry_threshold: float = 30.0
    exit_threshold: float = 15.0
    max_holding_bars: int = 336
    atr_trail_mult: float = 4.0
    kelly_fraction: float = 0.25
    max_risk_per_trade: float = 0.15
    score_exit_patience: int = 3
    n_bars: int = 2000
    grid_levels: int = 20
    grid_spacing: str = "geometric"
    grid_order_size: float = 100.0
    atr_multiplier: float = 3.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class CryptoBacktestResponse(BaseModel):
    sharpe: float
    sortino: float
    cagr: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    total_fees: float
    total_funding: float
    n_trades: int
    final_equity: float
    total_return_pct: float
    equity_curve: List[Dict[str, Any]]
    buy_hold_curve: List[Dict[str, Any]]
    price_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]] = []
    regimes_sampled: List[str]
    regime_counts: Dict[str, int]
    baselines: Dict[str, Dict[str, float]]
    score_reachability: Dict[str, Any]
    data_source: str = "live"
    start_date: str = ""
    end_date: str = ""
    n_bars_actual: int = 0

def _fetch_ohlcv(
    symbol: str, timeframe: str, since: datetime, until: datetime,
) -> pd.DataFrame:
    """Fetch OHLCV from the exchange adapter.  Raises on failure."""
    from crypto.adapters.price_adapter import CryptoPriceAdapter, EXCHANGE_CONFIGS
    adapter = CryptoPriceAdapter(EXCHANGE_CONFIGS["binance"])
    df = adapter.fetch(symbol, timeframe, since, until)
    if df is None or df.empty:
        raise RuntimeError(f"No OHLCV data returned for {symbol} ({timeframe})")
    return df

def _fetch_funding(
    symbol: str, since: datetime, until: datetime,
) -> Optional[pd.Series]:
    """Fetch funding rates. Returns None on failure (non-critical)."""
    try:
        from crypto.adapters.price_adapter import CryptoPriceAdapter, EXCHANGE_CONFIGS
        adapter = CryptoPriceAdapter(EXCHANGE_CONFIGS["binance"])
        df = adapter.fetch_funding_rates(symbol, since, until)
        if df is not None and not df.empty and "fundingRate" in df.columns:
            return df["fundingRate"]
    except Exception as e:
        logger.info("Funding rate fetch failed for %s (non-critical): %s", symbol, e)
    return None

def _sample_equity(eq: pd.Series, target_points: int = 500) -> List[Dict[str, Any]]:
    """Downsample equity curve to ~target_points, always including last point."""
    if len(eq) == 0:
        return []
    step = max(1, len(eq) // target_points)
    pts = []
    for i in range(0, len(eq), step):
        pts.append({"date": str(eq.index[i]), "equity": round(float(eq.iloc[i]), 2)})
    last_idx = len(eq) - 1
    if last_idx % step != 0:
        pts.append({"date": str(eq.index[last_idx]), "equity": round(float(eq.iloc[last_idx]), 2)})
    return pts

def _sample_price(close: pd.Series, eq: pd.Series, target_points: int = 500) -> List[Dict[str, Any]]:
    """Downsample the close price series to match equity curve sampling."""
    if len(close) == 0:
        return []
    offset = max(0, len(close) - len(eq))
    step = max(1, len(eq) // target_points)
    pts = []
    for i in range(0, len(eq), step):
        src_i = offset + i
        if src_i < len(close):
            pts.append({"date": str(close.index[src_i]), "price": round(float(close.iloc[src_i]), 2)})
    last_idx = len(eq) - 1
    last_src = offset + last_idx
    if last_idx % step != 0 and last_src < len(close):
        pts.append({"date": str(close.index[last_src]), "price": round(float(close.iloc[last_src]), 2)})
    return pts

def _sample_regimes(regimes: pd.Series, eq: pd.Series, target_points: int = 500) -> List[str]:
    """Downsample regime labels to match the equity curve sampling."""
    if len(regimes) == 0:
        return []
    step = max(1, len(eq) // target_points)
    labels = []
    for i in range(0, len(eq), step):
        if i < len(regimes):
            labels.append(str(regimes.iloc[i]))
        else:
            labels.append("RANGING")
    last_idx = len(eq) - 1
    if last_idx % step != 0:
        if last_idx < len(regimes):
            labels.append(str(regimes.iloc[last_idx]))
        else:
            labels.append("RANGING")
    return labels

def _serialize_trades(
    trades, regimes: pd.Series, max_trades: int = 500,
) -> List[Dict[str, Any]]:
    """Convert CryptoTradeRecords to JSON-safe dicts with regime context."""
    out = []
    for t in trades[-max_trades:]:
        d = t.to_dict()
        bar = t.bar_idx
        if 0 <= bar < len(regimes):
            d["regime"] = str(regimes.iloc[bar])
        else:
            d["regime"] = "RANGING"
        out.append(d)
    return out

@router.get("/markets")
async def get_markets(exchange: str = "binance"):
    """Return all available perpetual futures from exchange (cached)."""
    markets = await _get_markets(exchange)
    is_offline = not _market_cache.get("markets")
    categories = sorted(set(m.get("category", "Other") for m in markets))
    return {
        "markets": markets,
        "categories": categories,
        "offline": is_offline,
        "count": len(markets),
    }

@router.post("/backtest", response_model=CryptoBacktestResponse)
async def run_crypto_backtest(request: CryptoBacktestRequest):
    """Run a crypto bot backtest with the specified parameters."""
    try:
        from crypto.services.backtest_service import (
            CryptoBacktestConfig,
            CryptoBacktestService,
        )
        from crypto.regime.detector import CryptoRegimeConfig

        tf = request.timeframe
        bpd = bars_per_day(tf)

        if request.start_date and request.end_date:
            start_dt = datetime.fromisoformat(request.start_date)
            end_dt = datetime.fromisoformat(request.end_date)
            n_bars = _compute_n_bars(start_dt, end_dt, tf)
            n_bars = max(MIN_BARS, min(MAX_BARS, n_bars))
        else:
            n_bars = max(MIN_BARS, min(MAX_BARS, request.n_bars))
            default_range = _TF_DEFAULT_RANGE.get(tf, timedelta(days=90))
            end_dt = datetime.now(timezone.utc).replace(tzinfo=None)
            start_dt = end_dt - default_range

        if n_bars < MIN_BARS:
            raise HTTPException(
                status_code=400,
                detail=f"Date range too short: {n_bars} bars (minimum {MIN_BARS}). "
                       f"Widen the date range or use a smaller timeframe.",
            )

        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        try:
            ohlcv, funding_raw = await asyncio.gather(
                asyncio.to_thread(_fetch_ohlcv, request.symbol, tf, start_dt, end_dt),
                asyncio.to_thread(_fetch_funding, request.symbol, start_dt, end_dt),
            )
        except Exception as e:
            logger.error("Data fetch failed for %s: %s", request.symbol, e)
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch market data for {request.symbol}: {e}. "
                       f"Check that the symbol exists and the exchange is reachable.",
            )

        if len(ohlcv) < MIN_BARS:
            raise HTTPException(
                status_code=422,
                detail=f"Only {len(ohlcv)} bars available for {request.symbol} "
                       f"({tf}, {start_str} to {end_str}), need at least {MIN_BARS}. "
                       f"Widen the date range or use a smaller timeframe.",
            )

        n_bars = len(ohlcv)

        if funding_raw is not None and len(funding_raw) > 0:
            funding = funding_raw.reindex(ohlcv.index, method="ffill").fillna(0.0001)
        else:
            funding = pd.Series(0.0001, index=ohlcv.index)

        warmup = min(500, max(50, int(21 * bpd)))
        warmup = min(warmup, n_bars // 4)
        comp_win = min(120, max(20, int(10 * bpd)))
        ic_win = min(120, max(20, int(10 * bpd)))
        fund_win = min(200, max(20, int(14 * bpd)))

        bt_config = CryptoBacktestConfig(
            symbol=request.symbol,
            timeframe=tf,
            initial_capital=request.initial_capital,
            strategy_mode=request.strategy_mode,
            leverage=request.leverage,
            cost_preset=request.cost_preset,
            entry_threshold=request.entry_threshold,
            exit_threshold=request.exit_threshold,
            max_holding_bars=request.max_holding_bars,
            atr_trail_mult=request.atr_trail_mult,
            kelly_fraction=request.kelly_fraction,
            max_risk_per_trade=request.max_risk_per_trade,
            score_exit_patience=request.score_exit_patience,
            compression_window=comp_win,
            ic_window=ic_win,
            ic_horizon=min(20, max(3, comp_win // 6)),
            funding_window=fund_win,
            grid_levels=request.grid_levels,
            grid_spacing=request.grid_spacing,
            grid_order_size=request.grid_order_size,
            atr_multiplier=request.atr_multiplier,
            regime_config=CryptoRegimeConfig(
                warmup_bars=warmup,
                rolling_window=warmup,
                refit_every=max(50, warmup // 5),
                vol_window=min(96, max(10, warmup // 4)),
                cooldown_bars=3,
                circuit_breaker_dd=-0.25,
            ),
        )

        svc = CryptoBacktestService()
        result = await asyncio.to_thread(svc.run, ohlcv, bt_config, funding_rates=funding)

        eq = result["equity_curve"]
        equity_points = _sample_equity(eq)

        regimes = result.get("regimes", pd.Series(dtype=str))
        regime_labels = _sample_regimes(regimes, eq)
        regime_counts = dict(regimes.value_counts()) if len(regimes) > 0 else {}

        baselines = result.get("baselines", {})
        bh_data = baselines.get("buy_and_hold", {})
        bh_curve = []
        if bh_data and "final_equity" in bh_data:
            close = ohlcv["close"]
            first_valid = max(0, len(close) - len(eq) if len(eq) < len(close) else 0)
            valid_close = close.iloc[first_valid:]
            bh_returns = valid_close.pct_change().fillna(0)
            bh_equity = (1 + bh_returns).cumprod() * request.initial_capital
            bh_curve = _sample_equity(bh_equity)

        close_series = ohlcv["close"]
        price_curve = _sample_price(close_series, eq)

        raw_trades = result.get("trades", [])
        trade_dicts = _serialize_trades(raw_trades, regimes)

        return CryptoBacktestResponse(
            sharpe=round(result["sharpe"], 4),
            sortino=round(result["sortino"], 4),
            cagr=round(result["cagr"], 4),
            max_drawdown=round(result["max_drawdown"], 4),
            calmar=round(result["calmar"], 4),
            win_rate=round(result["win_rate"], 4),
            profit_factor=round(min(result["profit_factor"], 999), 4),
            avg_trade_pnl=round(result["avg_trade_pnl"], 2),
            total_fees=round(result["total_fees"], 2),
            total_funding=round(result["total_funding"], 2),
            n_trades=result["n_trades"],
            final_equity=round(result["final_equity"], 2),
            total_return_pct=round(result["total_return_pct"], 2),
            equity_curve=equity_points,
            buy_hold_curve=bh_curve,
            price_curve=price_curve,
            trades=trade_dicts,
            regimes_sampled=regime_labels,
            regime_counts={str(k): int(v) for k, v in regime_counts.items()},
            baselines=baselines,
            score_reachability=result.get("score_reachability", {}),
            data_source="live",
            start_date=start_str,
            end_date=end_str,
            n_bars_actual=len(ohlcv),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Crypto backtest error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config/defaults")
async def get_defaults():
    """Return default crypto bot configuration."""
    return {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "1h",
        "initial_capital": 10_000.0,
        "strategy_mode": "adaptive",
        "leverage": 3.0,
        "entry_threshold": 30.0,
        "exit_threshold": 15.0,
        "max_holding_bars": 336,
        "atr_trail_mult": 4.0,
        "kelly_fraction": 0.25,
        "max_risk_per_trade": 0.15,
        "score_exit_patience": 3,
        "grid_levels": 20,
        "grid_spacing": "geometric",
        "grid_order_size": 100.0,
        "atr_multiplier": 3.0,
    }

@router.get("/strategies")
async def get_strategies():
    """Return available strategy modes."""
    return {
        "strategies": [
            {"id": "directional", "name": "Directional Futures",
             "description": "Trend/mean-reversion scoring with leveraged futures"},
            {"id": "grid", "name": "Grid Trading",
             "description": "Range-bound market-making with geometric grid"},
            {"id": "adaptive", "name": "Adaptive (Recommended)",
             "description": "Auto-switches between Directional and Grid based on regime"},
        ]
    }
