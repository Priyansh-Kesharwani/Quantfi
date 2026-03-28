import React from 'react';
import { Play, AlertTriangle, Clock } from 'lucide-react';

const TEMPLATES = {
  conservative: { label: 'CONSERVATIVE', desc: 'Tight stops, high entry bar, fewer positions', color: '#22C55E', entryThreshold: 80, maxPositions: 6, minInvestedFraction: 0.3, scoringMode: 'mean_reversion', simulationMode: 'tactical' },
  balanced:     { label: 'BALANCED',     desc: 'Adaptive regime-switching strategy',           color: '#6366F1', entryThreshold: 70, maxPositions: 10, minInvestedFraction: 0.2, scoringMode: 'adaptive', simulationMode: 'tactical' },
  aggressive:   { label: 'AGGRESSIVE',   desc: 'Adaptive, loose stops, more slots',            color: '#EF4444', entryThreshold: 60, maxPositions: 15, minInvestedFraction: 0.0, scoringMode: 'adaptive', simulationMode: 'tactical' },
  allocation:   { label: 'ALLOCATION',   desc: 'Regime-aware always-invested allocation engine', color: '#F59E0B', simulationMode: 'allocation', riskOnPct: 0.95, riskOffPct: 0.60, thetaTilt: 0.0, rebalanceFreq: 42, scoringMode: 'adaptive' },
};

const SimulationForm = ({
  assetList,
  selectedAssets,
  template,
  config,
  running,
  error,
  result,
  effectiveMaxPositions,
  onToggleAsset,
  onTemplateChange,
  onUpdateConfig,
  onRunSimulation,
}) => {
  return (
    <div className="glass-effect rounded-sm p-6" data-testid="sim-config">
      <h2 className="text-xl font-bold mb-6">CONFIGURATION</h2>
      <div className="space-y-4">

        {/* Asset Selection */}
        <div>
          <label className="text-sm text-muted-foreground mb-2 block">SELECT ASSETS</label>
          <div className="flex flex-wrap gap-2 max-h-32 overflow-y-auto">
            {assetList.map(asset => (
              <button
                key={asset.symbol}
                onClick={() => onToggleAsset(asset.symbol)}
                className={`px-3 py-1.5 rounded text-xs font-bold transition ${
                  selectedAssets.includes(asset.symbol)
                    ? 'bg-primary text-primary-foreground'
                    : 'glass-effect text-muted-foreground hover:text-foreground'
                }`}
              >
                {asset.symbol}
              </button>
            ))}
          </div>
          <p className="text-[10px] text-muted-foreground mt-1">{selectedAssets.length} selected</p>
        </div>

        {/* Strategy Template */}
        <div>
          <label className="text-sm text-muted-foreground mb-2 block">STRATEGY TEMPLATE</label>
          <div className="grid grid-cols-3 gap-2">
            {Object.entries(TEMPLATES).map(([key, tpl]) => (
              <button
                key={key}
                onClick={() => onTemplateChange(key)}
                className={`p-2 rounded text-center transition border ${
                  template === key
                    ? 'border-primary bg-primary/10'
                    : 'border-white/10 glass-effect hover:bg-white/5'
                }`}
              >
                <div className="text-[10px] font-bold" style={{ color: tpl.color }}>{tpl.label}</div>
              </button>
            ))}
          </div>
          <p className="text-[10px] text-muted-foreground mt-1">
            {TEMPLATES[template]?.desc}
          </p>
        </div>

        {/* Date Range */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-[10px] text-muted-foreground mb-1 block">START DATE</label>
            <input type="date" value={config.startDate}
              onChange={e => onUpdateConfig('startDate', e.target.value)}
              className="w-full p-2 glass-effect rounded text-xs font-data" />
          </div>
          <div>
            <label className="text-[10px] text-muted-foreground mb-1 block">END DATE</label>
            <input type="date" value={config.endDate}
              onChange={e => onUpdateConfig('endDate', e.target.value)}
              className="w-full p-2 glass-effect rounded text-xs font-data" />
          </div>
        </div>
        <div className="flex gap-2">
          {[3, 5, 10].map(y => (
            <button key={y} onClick={() => {
              const d = new Date();
              d.setFullYear(d.getFullYear() - y);
              onUpdateConfig('startDate', d.toISOString().split('T')[0]);
            }} className="px-2 py-1 text-xs glass-effect rounded hover:bg-white/10 transition">
              {y}Y
            </button>
          ))}
        </div>

        {/* Capital */}
        <div>
          <label className="text-[10px] text-muted-foreground mb-1 block">INITIAL CAPITAL ($)</label>
          <input type="number" value={config.initialCapital}
            onChange={e => onUpdateConfig('initialCapital', e.target.value)}
            className="w-full p-2 glass-effect rounded text-xs font-data" />
        </div>

        {config.simulationMode === 'allocation' ? (
          <>
            <div>
              <label className="text-[10px] text-muted-foreground mb-1 block">
                RISK-ON ALLOCATION ({Math.round(config.riskOnPct * 100)}%)
              </label>
              <input type="range" min="0.60" max="1.00" step="0.05"
                value={config.riskOnPct}
                onChange={e => onUpdateConfig('riskOnPct', parseFloat(e.target.value))}
                className="w-full" />
              <p className="text-[10px] text-muted-foreground">
                Equity allocation during risk-on regimes
              </p>
            </div>

            <div>
              <label className="text-[10px] text-muted-foreground mb-1 block">
                RISK-OFF ALLOCATION ({Math.round(config.riskOffPct * 100)}%)
              </label>
              <input type="range" min="0.20" max="0.90" step="0.05"
                value={config.riskOffPct}
                onChange={e => onUpdateConfig('riskOffPct', parseFloat(e.target.value))}
                className="w-full" />
              <p className="text-[10px] text-muted-foreground">
                Equity allocation during risk-off regimes
              </p>
            </div>

            <div>
              <label className="text-[10px] text-muted-foreground mb-1 block">
                TILT INTENSITY ({config.thetaTilt})
              </label>
              <input type="range" min="0.0" max="2.0" step="0.1"
                value={config.thetaTilt}
                onChange={e => onUpdateConfig('thetaTilt', parseFloat(e.target.value))}
                className="w-full" />
              <p className="text-[10px] text-muted-foreground">
                Higher = more weight on top-scored assets (0 = equal weight)
              </p>
            </div>

            <div>
              <label className="text-[10px] text-muted-foreground mb-1 block">
                REBALANCE FREQUENCY ({config.rebalanceFreq} days)
              </label>
              <input type="range" min="21" max="252" step="21"
                value={config.rebalanceFreq}
                onChange={e => onUpdateConfig('rebalanceFreq', parseInt(e.target.value))}
                className="w-full" />
              <p className="text-[10px] text-muted-foreground">
                Scheduled rebalance interval (also triggers on regime shift)
              </p>
            </div>

            {selectedAssets.length < 3 && (
              <div className="p-2 rounded border border-yellow-500/30 bg-yellow-500/5">
                <p className="text-[10px] text-yellow-400">
                  Allocation engine works best with 3+ diversified assets (e.g. add TLT, GLD)
                </p>
              </div>
            )}
          </>
        ) : (
          <>
            <div>
              <label className="text-[10px] text-muted-foreground mb-1 block">
                ENTRY SCORE THRESHOLD ({config.entryThreshold})
              </label>
              <input type="range" min="40" max="90" step="5"
                value={config.entryThreshold}
                onChange={e => onUpdateConfig('entryThreshold', e.target.value)}
                className="w-full" />
              <p className="text-[10px] text-muted-foreground">
                Enter when composite score ≥ {config.entryThreshold}
              </p>
            </div>

            <div>
              <label className="text-[10px] text-muted-foreground mb-1 block">MAX POSITIONS</label>
              <input type="number" value={config.maxPositions} min="1" max="20"
                onChange={e => onUpdateConfig('maxPositions', e.target.value)}
                className="w-full p-2 glass-effect rounded text-xs font-data" />
              {selectedAssets.length > 0 && effectiveMaxPositions < parseInt(config.maxPositions) && (
                <p className="text-[10px] text-yellow-400/80 mt-1">
                  Capped at {effectiveMaxPositions} by selected assets ({selectedAssets.length})
                </p>
              )}
              {selectedAssets.length === 1 && (
                <p className="text-[10px] text-muted-foreground mt-1">
                  Single-asset sim — only 1 position held at a time
                </p>
              )}
            </div>

            <div>
              <label className="text-[10px] text-muted-foreground mb-1 block">
                MIN INVESTED ({Math.round(config.minInvestedFraction * 100)}%)
              </label>
              <input type="range" min="0" max="0.7" step="0.05"
                value={config.minInvestedFraction}
                onChange={e => onUpdateConfig('minInvestedFraction', parseFloat(e.target.value))}
                className="w-full" />
              <p className="text-[10px] text-muted-foreground">
                {config.minInvestedFraction > 0
                  ? `Always keep ≥${Math.round(config.minInvestedFraction * 100)}% invested (core-satellite)`
                  : 'Pure tactical — can go 100% cash'}
              </p>
            </div>
          </>
        )}

        {/* Scoring Mode */}
        <div>
          <label className="text-[10px] text-muted-foreground mb-2 block">SCORING MODE</label>
          <div className="grid grid-cols-3 gap-1">
            {[
              { key: 'mean_reversion', label: 'MEAN REVERT' },
              { key: 'adaptive', label: 'ADAPTIVE' },
              { key: 'trend_following', label: 'TREND' },
            ].map(m => (
              <button key={m.key}
                onClick={() => onUpdateConfig('scoringMode', m.key)}
                className={`px-2 py-1.5 rounded text-[10px] font-bold transition border ${
                  config.scoringMode === m.key
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-white/10 glass-effect text-muted-foreground hover:text-foreground'
                }`}
              >{m.label}</button>
            ))}
          </div>
          <p className="text-[10px] text-muted-foreground mt-1">
            {config.scoringMode === 'adaptive' ? 'HMM regime detection switches between strategies' :
             config.scoringMode === 'trend_following' ? 'Trend-following signals for bull markets' :
             'Classic mean-reversion dip-buying'}
          </p>
        </div>

        {/* Run Button */}
        <button onClick={onRunSimulation} disabled={running}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary text-primary-foreground rounded font-medium hover:bg-primary/90 transition disabled:opacity-50"
          data-testid="run-sim-btn">
          <Play className="w-5 h-5" />
          {running ? 'SIMULATING…' : 'RUN SIMULATION'}
        </button>

        {error && (
          <div className="flex items-start gap-2 p-3 rounded bg-destructive/10 text-destructive text-xs">
            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {result && (
          <div className="text-[10px] text-muted-foreground flex items-center gap-2">
            <Clock className="w-3 h-3" />
            Computed in {result.computation_time_s}s · {result.data_range}
          </div>
        )}
      </div>
    </div>
  );
};

export { TEMPLATES };
export default SimulationForm;
