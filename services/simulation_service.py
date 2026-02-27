from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime, timezone
import importlib.util
import sys


class SimulationService:
    def __init__(
        self,
        fx_provider: Any,
        config: Any,
        project_root: Path,
    ) -> None:
        self._fx_provider = fx_provider
        self._config = config
        self._project_root = Path(project_root)
        self._fx_fallback = getattr(config, "fx_fallback_usd_inr", 83.5)

    def _load_simulator_module(self) -> Any:
        sim_path = str(
            self._project_root / "backtester" / "portfolio_simulator.py"
        )
        spec = importlib.util.spec_from_file_location(
            "portfolio_simulator", sim_path
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["portfolio_simulator"] = mod
        spec.loader.exec_module(mod)
        return mod

    def get_templates(self) -> Dict[str, Any]:
        return {
            "conservative": {
                "entry_score_threshold": 80.0,
                "exit_config": {
                    "atr_init_mult": 1.5,
                    "atr_trail_mult": 2.0,
                    "min_stop_pct": 5.0,
                    "score_rel_mult": 0.5,
                    "score_abs_floor": 40.0,
                    "max_holding_days": 20,
                },
                "max_positions": 6,
                "use_score_weighting": True,
                "slippage_bps": 5.0,
            },
            "balanced": {
                "entry_score_threshold": 70.0,
                "exit_config": {
                    "atr_init_mult": 2.0,
                    "atr_trail_mult": 2.5,
                    "min_stop_pct": 4.0,
                    "score_rel_mult": 0.4,
                    "score_abs_floor": 35.0,
                    "max_holding_days": 30,
                },
                "max_positions": 10,
                "use_score_weighting": True,
                "slippage_bps": 5.0,
            },
            "aggressive": {
                "entry_score_threshold": 60.0,
                "exit_config": {
                    "atr_init_mult": 2.5,
                    "atr_trail_mult": 3.0,
                    "min_stop_pct": 3.0,
                    "score_rel_mult": 0.3,
                    "score_abs_floor": 30.0,
                    "max_holding_days": 45,
                },
                "max_positions": 15,
                "use_score_weighting": True,
                "slippage_bps": 5.0,
            },
        }

    def get_cost_presets(self) -> Dict[str, Any]:
        mod = self._load_simulator_module()
        return {"cost_presets": mod.COST_PRESETS}

    async def run_simulation(
        self,
        db: Any,
        request: Any,
        logger: Any,
    ) -> Dict[str, Any]:
        mod = self._load_simulator_module()
        tpl = self.get_templates().get(request.template or "", {})
        exit_cfg_dict = (
            request.exit_config.model_dump()
            if request.exit_config
            else tpl.get("exit_config", {})
        )
        exit_params = (
            mod.ExitParams(**exit_cfg_dict)
            if exit_cfg_dict
            else mod.ExitParams()
        )
        symbols = [s.upper().strip() for s in request.symbols if s.strip()]
        if not symbols:
            raise ValueError("At least one symbol is required")

        start_dt = datetime.fromisoformat(request.start_date)
        end_dt = datetime.fromisoformat(request.end_date)
        cost_free = getattr(request, "cost_free", False)

        config = mod.SimConfig(
            symbols=symbols,
            start_date=start_dt,
            end_date=end_dt,
            initial_capital=request.initial_capital,
            entry_score_threshold=request.entry_score_threshold,
            exit_params=exit_params,
            max_positions=request.max_positions,
            use_score_weighting=tpl.get(
                "use_score_weighting", request.use_score_weighting
            ),
            min_position_notional=request.min_position_notional,
            slippage_bps=0.0
            if cost_free
            else tpl.get("slippage_bps", request.slippage_bps),
            cost_free=cost_free,
            run_benchmarks=request.run_benchmarks,
        )

        # #region agent log
        try:
            import json as _json, time as _time
            _log_path = str(self._project_root / ".cursor" / "debug-9f2122.log")
            _payload = _json.dumps({"sessionId":"9f2122","hypothesisId":"H1","location":"simulation_service.py:config_build","message":"SimConfig created","data":{"request_max_positions":request.max_positions,"config_max_positions":config.max_positions,"request_entry_threshold":request.entry_score_threshold,"template":request.template,"exit_max_holding_days":exit_params.max_holding_days,"tpl_keys":list(tpl.keys())},"timestamp":int(_time.time()*1000)})
            with open(_log_path, "a") as _f: _f.write(_payload + "\n")
        except Exception: pass
        # #endregion

        asset_meta = {}
        for sym in symbols:
            doc = await db.assets.find_one({"symbol": sym}, {"_id": 0})
            if doc:
                asset_meta[sym] = {
                    "asset_type": doc.get("asset_type", "equity"),
                    "currency": doc.get("currency", "USD"),
                    "exchange": doc.get("exchange"),
                }
            else:
                asset_meta[sym] = {
                    "asset_type": "equity",
                    "currency": "USD",
                }

        usd_inr = (
            self._fx_provider.fetch_usd_inr_rate() or self._fx_fallback
        )
        logger.info(
            "Simulation: %s | %s -> %s | capital=%s",
            symbols,
            start_dt.date(),
            end_dt.date(),
            config.initial_capital,
        )

        date_index, assets_data = mod.prepare_multi_asset_data(
            symbols=symbols,
            start_date=start_dt,
            end_date=end_dt,
            asset_meta=asset_meta,
            usd_inr_rate=usd_inr,
        )
        simulator = mod.PortfolioSimulator(config)
        result = simulator.run(date_index, assets_data)

        if getattr(request, "export_trades", False) and result.get("trades"):
            run_id = getattr(request, "run_id", None) or str(uuid.uuid4())[:8]
            debug_dir = self._project_root / "validation" / "debug"
            csv_path = debug_dir / f"trades_{run_id}.csv"
            try:
                written = mod.export_trades_csv(
                    str(csv_path),
                    trades=result["trades"],
                    run_id=run_id,
                )
                result["trades_export_path"] = written
            except Exception as export_err:
                logger.warning("Trade export failed: %s", export_err)

        try:
            store_doc = {
                "symbols": symbols,
                "config": config.to_dict(),
                "total_return_pct": result["total_return_pct"],
                "sharpe_ratio": result["sharpe_ratio"],
                "max_drawdown_pct": result["max_drawdown_pct"],
                "total_trades": result["total_trades"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            await db.simulation_results.insert_one(store_doc)
        except Exception as db_err:
            logger.warning("Failed to save simulation to DB: %s", db_err)

        return result
