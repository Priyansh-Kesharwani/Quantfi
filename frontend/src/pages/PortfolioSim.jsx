import React, { useState, useEffect } from 'react';

import { useWatchlist } from '@/contexts/WatchlistContext';
import SimulationForm, { TEMPLATES } from '@/components/simulation/SimulationForm';
import ResultsPanel from '@/components/simulation/ResultsPanel';

import api from '@/api';
import { toast } from 'sonner';

const PortfolioSim = () => {
  const { assetList, loading: wlLoading } = useWatchlist();
  const [selectedAssets, setSelectedAssets] = useState([]);
  const [template, setTemplate] = useState('balanced');
  const [config, setConfig] = useState({
    startDate: '2015-01-01',
    endDate: new Date().toISOString().split('T')[0],
    initialCapital: 100000,
    entryThreshold: 70,
    maxPositions: 10,
    maxHoldingDays: 30,
    slippageBps: 5,
    minInvestedFraction: 0.2,
    scoringMode: 'adaptive',
    simulationMode: 'tactical',
    riskOnPct: 0.95,
    riskOffPct: 0.60,
    thetaTilt: 0.0,
    rebalanceFreq: 42,
  });
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [resultStale, setResultStale] = useState(false);

  useEffect(() => {
    if (assetList.length > 0 && selectedAssets.length === 0) {
      setSelectedAssets(assetList.slice(0, Math.min(5, assetList.length)).map(a => a.symbol));
    }
  }, [assetList, selectedAssets.length]);

  const handleTemplateChange = (key) => {
    const t = TEMPLATES[key];
    setTemplate(key);
    setConfig(prev => ({
      ...prev,
      entryThreshold: t.entryThreshold ?? prev.entryThreshold,
      maxPositions: t.maxPositions ?? prev.maxPositions,
      minInvestedFraction: t.minInvestedFraction ?? prev.minInvestedFraction,
      scoringMode: t.scoringMode ?? prev.scoringMode,
      simulationMode: t.simulationMode ?? 'tactical',
      riskOnPct: t.riskOnPct ?? prev.riskOnPct,
      riskOffPct: t.riskOffPct ?? prev.riskOffPct,
      thetaTilt: t.thetaTilt ?? prev.thetaTilt,
      rebalanceFreq: t.rebalanceFreq ?? prev.rebalanceFreq,
    }));
    if (result) setResultStale(true);
  };

  const updateConfig = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }));
    if (result) setResultStale(true);
  };

  const toggleAsset = (sym) => {
    setSelectedAssets(prev =>
      prev.includes(sym) ? prev.filter(s => s !== sym) : [...prev, sym]
    );
    if (result) setResultStale(true);
  };

  const effectiveMaxPositions = Math.min(
    parseInt(config.maxPositions) || 1,
    selectedAssets.length || 1,
  );

  const runSimulation = async () => {
    if (selectedAssets.length === 0) {
      toast.error('Select at least one asset');
      return;
    }
    setRunning(true);
    setError(null);
    try {
      const payload = {
        symbols: selectedAssets,
        start_date: config.startDate + 'T00:00:00',
        end_date: config.endDate + 'T00:00:00',
        initial_capital: parseFloat(config.initialCapital),
        entry_score_threshold: parseFloat(config.entryThreshold),
        max_positions: parseInt(config.maxPositions),
        slippage_bps: parseFloat(config.slippageBps),
        run_benchmarks: true,
        template: template,
        min_invested_fraction: parseFloat(config.minInvestedFraction),
        scoring_mode: config.scoringMode,
        simulation_mode: config.simulationMode,
        risk_on_equity_pct: parseFloat(config.riskOnPct),
        risk_off_equity_pct: parseFloat(config.riskOffPct),
        theta_tilt: parseFloat(config.thetaTilt),
        rebalance_freq_days: parseInt(config.rebalanceFreq),
      };
      const response = await api.runSimulation(payload);
      setResult(response.data);
      setResultStale(false);
      toast.success(`Simulation complete — ${response.data.total_trades} trades`);
    } catch (err) {
      const detail = err?.response?.data?.detail || 'Simulation failed';
      setError(detail);
      toast.error(detail);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="p-6 lg:p-8 max-w-[1600px] mx-auto" data-testid="portfolio-sim-page">
      <div className="mb-8">
        <h1 className="text-4xl font-bold tracking-tight mb-2">PORTFOLIO SIMULATION</h1>
        <p className="text-muted-foreground">
          Adaptive backtesting with HMM regime detection and dynamic entry/exit signals — real market data
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <SimulationForm
          assetList={assetList}
          selectedAssets={selectedAssets}
          template={template}
          config={config}
          running={running}
          error={error}
          result={result}
          effectiveMaxPositions={effectiveMaxPositions}
          onToggleAsset={toggleAsset}
          onTemplateChange={handleTemplateChange}
          onUpdateConfig={updateConfig}
          onRunSimulation={runSimulation}
        />

        <div className="lg:col-span-2">
          <ResultsPanel
            result={result}
            config={config}
            resultStale={resultStale}
          />
        </div>
      </div>
    </div>
  );
};

export default PortfolioSim;
