from backend.app_config import get_backend_config
from infrastructure.adapters.price_adapter import PriceAdapter
from infrastructure.adapters.fx_adapter import FXAdapter
from infrastructure.adapters.news_adapter import NewsAdapter

_price_adapter: PriceAdapter | None = None
_fx_adapter: FXAdapter | None = None
_news_adapter: NewsAdapter | None = None


def _get_price_adapter() -> PriceAdapter:
    global _price_adapter
    if _price_adapter is None:
        _price_adapter = PriceAdapter(get_backend_config())
    return _price_adapter


def _get_fx_adapter() -> FXAdapter:
    global _fx_adapter
    if _fx_adapter is None:
        _fx_adapter = FXAdapter(get_backend_config())
    return _fx_adapter


def _get_news_adapter() -> NewsAdapter:
    global _news_adapter
    if _news_adapter is None:
        _news_adapter = NewsAdapter(get_backend_config())
    return _news_adapter


class PriceProvider:
    @classmethod
    def fetch_historical_data(
        cls,
        symbol: str,
        period: str = "2y",
        exchange=None,
        start_date=None,
        end_date=None,
    ):
        return _get_price_adapter().fetch_historical_data(
            symbol, period, exchange, start_date, end_date
        )

    @classmethod
    def fetch_latest_price(cls, symbol: str, exchange=None):
        return _get_price_adapter().fetch_latest_price(symbol, exchange)


class FXProvider:
    @classmethod
    def fetch_usd_inr_rate(cls):
        return _get_fx_adapter().fetch_usd_inr_rate()


class NewsProvider:
    def __init__(self):
        self._adapter = _get_news_adapter()

    def fetch_latest_news(self, query=None, page_size=None):
        return self._adapter.fetch_latest_news(query or "", page_size)

    def fetch_news_for_assets(self, symbols, per_asset=8):
        return self._adapter.fetch_news_for_assets(symbols, per_asset)
