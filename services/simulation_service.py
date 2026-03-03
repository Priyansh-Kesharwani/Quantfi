from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime, timezone

from domain.protocols import IFXProvider


class SimulationService:
    def __init__(
        self,
        fx_provider: IFXProvider,
        config: Any,
        project_root: Path,
    ) -> None:
        self._fx_provider = fx_provider
        self._config = config
        self._project_root = Path(project_root)
        self._fx_fallback = getattr(config, "fx_fallback_usd_inr", 83.5)

    @staticmethod
    def _load_simulator_module() -> Any:
        import backtester.portfolio_simulator as mod
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
                "min_invested_fraction": 0.3,
                "scoring_mode": "mean_reversion",
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
                "min_invested_fraction": 0.2,
                "scoring_mode": "adaptive",
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
                "min_invested_fraction": 0.0,
                "scoring_mode": "adaptive",
            },
            "allocation": {
                "simulation_mode": "allocation",
                "risk_on_equity_pct": 0.95,
                "risk_off_equity_pct": 0.60,
                "theta_tilt": 0.0,
                "rebalance_freq_days": 42,
                "min_rebalance_delta": 0.03,
                "jump_penalty": 25.0,
                "drawdown_circuit_threshold": -0.15,
                "cash_return_annual": 0.02,
                "slippage_bps": 5.0,
                "scoring_mode": "adaptive",
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

        symbols = [s.upper().strip() for s in request.symbols if s.strip()]
        if not symbols:
            raise ValueError("At least one symbol is required")

        start_dt = datetime.fromisoformat(request.start_date)
        end_dt = datetime.fromisoformat(request.end_date)
        cost_free = getattr(request, "cost_free", False)
        scoring_mode = getattr(request, "scoring_mode", tpl.get("scoring_mode", "adaptive"))

        sim_mode = getattr(request, "simulation_mode", tpl.get("simulation_mode", "tactical"))

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
                asset_meta[sym] = {"asset_type": "equity", "currency": "USD"}

        usd_inr = self._fx_provider.fetch_usd_inr_rate() or self._fx_fallback

        logger.info(
            "Simulation [%s]: %s | %s -> %s | capital=%s",
            sim_mode, symbols, start_dt.date(), end_dt.date(),
            request.initial_capital,
        )

        date_index, assets_data = mod.prepare_multi_asset_data(
            symbols=symbols,
            start_date=start_dt,
            end_date=end_dt,
            asset_meta=asset_meta,
            usd_inr_rate=usd_inr,
            scoring_mode=scoring_mode,
        )

        if sim_mode == "allocation":
            alloc_cfg = mod.AllocationConfig(
                symbols=symbols,
                start_date=start_dt,
                end_date=end_dt,
                initial_capital=request.initial_capital,
                risk_on_equity_pct=getattr(request, "risk_on_equity_pct", tpl.get("risk_on_equity_pct", 0.85)),
                risk_off_equity_pct=getattr(request, "risk_off_equity_pct", tpl.get("risk_off_equity_pct", 0.40)),
                theta_tilt=getattr(request, "theta_tilt", tpl.get("theta_tilt", 2.0)),
                rebalance_freq_days=getattr(request, "rebalance_freq_days", tpl.get("rebalance_freq_days", 21)),
                min_rebalance_delta=getattr(request, "min_rebalance_delta", tpl.get("min_rebalance_delta", 0.05)),
                jump_penalty=getattr(request, "jump_penalty", tpl.get("jump_penalty", 25.0)),
                drawdown_circuit_threshold=getattr(request, "drawdown_circuit_threshold", tpl.get("drawdown_circuit_threshold", -0.10)),
                cash_return_annual=getattr(request, "cash_return_annual", tpl.get("cash_return_annual", 0.02)),
                slippage_bps=0.0 if cost_free else getattr(request, "slippage_bps", tpl.get("slippage_bps", 5.0)),
                cost_free=cost_free,
                scoring_mode=scoring_mode,
                run_benchmarks=request.run_benchmarks,
            )
            engine = mod.AllocationEngine(alloc_cfg)
            result = engine.run(date_index, assets_data)
            config_dict = alloc_cfg.to_dict()
        else:
            exit_cfg_dict = (
                request.exit_config.model_dump()
                if request.exit_config
                else tpl.get("exit_config", {})
            )
            exit_params = mod.ExitParams(**exit_cfg_dict) if exit_cfg_dict else mod.ExitParams()
            min_inv = getattr(request, "min_invested_fraction", tpl.get("min_invested_fraction", 0.0))

            config = mod.SimConfig(
                symbols=symbols,
                start_date=start_dt,
                end_date=end_dt,
                initial_capital=request.initial_capital,
                entry_score_threshold=request.entry_score_threshold,
                exit_params=exit_params,
                max_positions=request.max_positions,
                use_score_weighting=tpl.get("use_score_weighting", request.use_score_weighting),
                min_position_notional=request.min_position_notional,
                slippage_bps=0.0 if cost_free else tpl.get("slippage_bps", request.slippage_bps),
                cost_free=cost_free,
                run_benchmarks=request.run_benchmarks,
                min_invested_fraction=float(min_inv),
                scoring_mode=scoring_mode,
            )
            simulator = mod.PortfolioSimulator(config)
            result = simulator.run(date_index, assets_data)
            config_dict = config.to_dict()

        if getattr(request, "export_trades", False) and result.get("trades"):
            run_id = getattr(request, "run_id", None) or str(uuid.uuid4())[:8]
            debug_dir = self._project_root / "validation" / "debug"
            csv_path = debug_dir / f"trades_{run_id}.csv"
            try:
                written = mod.export_trades_csv(str(csv_path), trades=result["trades"], run_id=run_id)
                result["trades_export_path"] = written
            except Exception as export_err:
                logger.warning("Trade export failed: %s", export_err)

        try:
            store_doc = {
                "symbols": symbols,
                "config": config_dict,
                "total_return_pct": result.get("total_return_pct", 0),
                "sharpe_ratio": result.get("sharpe_ratio", 0),
                "max_drawdown_pct": result.get("max_drawdown_pct", 0),
                "total_trades": result.get("total_trades", 0),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            await db.simulation_results.insert_one(store_doc)
        except Exception as db_err:
            logger.warning("Failed to save simulation to DB: %s", db_err)

        return result
