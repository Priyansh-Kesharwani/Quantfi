"""Symbol resolution for crypto perpetual futures."""

from __future__ import annotations


def resolve_symbol(
    base: str,
    quote: str = "USDT",
    instrument: str = "perpetual",
) -> str:
    """Convert a base/quote pair into a CCXT-compatible symbol.

    Examples:
        resolve_symbol("BTC") -> "BTC/USDT:USDT"
        resolve_symbol("ETH", "BUSD", "spot") -> "ETH/BUSD"
    """
    base = base.upper().strip()
    quote = quote.upper().strip()
    if instrument == "perpetual":
        return f"{base}/{quote}:{quote}"
    return f"{base}/{quote}"


def parse_symbol(symbol: str) -> dict:
    """Parse a CCXT symbol back into components.

    Returns dict with keys: base, quote, instrument.
    """
    if ":" in symbol:
        pair, settle = symbol.split(":", 1)
        base, quote = pair.split("/", 1)
        return {"base": base, "quote": quote, "settle": settle, "instrument": "perpetual"}
    base, quote = symbol.split("/", 1)
    return {"base": base, "quote": quote, "settle": None, "instrument": "spot"}
