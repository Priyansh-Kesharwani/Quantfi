from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any
from functools import lru_cache

import yaml


class ConfigLoader:
    def __init__(self, config_dir: Path) -> None:
        self._config_dir = Path(config_dir)

    def load_yaml(self, relative_path: str) -> Dict[str, Any]:
        path = self._config_dir / relative_path
        if not path.exists():
            return {}
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def load_backend_section(self) -> Dict[str, Any]:
        data = self.load_yaml("phase1.yml")
        result = data.get("backend", {})
        dp = data.get("data_providers", {})
        if dp.get("symbol_aliases"):
            result.setdefault("symbol_aliases", dp["symbol_aliases"])
        if dp.get("exchange_suffixes"):
            result.setdefault("exchange_suffixes", dp["exchange_suffixes"])
        if dp.get("news_rss_base"):
            result.setdefault("news_rss_base", dp["news_rss_base"])
        if dp.get("news_rss_business"):
            result.setdefault("news_rss_business", dp["news_rss_business"])
        if dp.get("news_per_asset"):
            result.setdefault("news_per_asset", dp["news_per_asset"])
        return result


def get_config_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "config"


@lru_cache(maxsize=1)
def get_config_loader() -> ConfigLoader:
    return ConfigLoader(get_config_dir())
