import React, { useState, useEffect } from 'react';
import api from '../api';
import { formatCurrency, formatPercent, formatNumber } from '../utils';
import { Play, TrendingUp, DollarSign, Calendar } from 'lucide-react';
import { toast } from 'sonner';

const BacktestLab = () => {
  const [assets, setAssets] = useState([]);
  const [config, setConfig] = useState({
    symbol: '',
    startDate: new Date(Date.now() - 365 * 2 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    endDate: new Date().toISOString().split('T')[0],
    dcaAmount: 5000,
    dcaCadence: 'monthly',
    buyDipThreshold: 60
  });
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    fetchAssets();
  }, []);

  const fetchAssets = async () => {
    try {
      const response = await api.getAssets();
      setAssets(response.data);
      if (response.data.length > 0) {
        setConfig(prev => ({ ...prev, symbol: response.data[0].symbol }));
      }
    } catch (error) {
      console.error('Error fetching assets:', error);
    }
  };

  const runBacktest = async () => {
    if (!config.symbol) {
      toast.error('Please select an asset');
      return;
    }

    setRunning(true);
    try {
      const response = await api.runBacktest({
        symbol: config.symbol,
        start_date: config.startDate + 'T00:00:00',
        end_date: config.endDate + 'T00:00:00',
        dca_amount: parseFloat(config.dcaAmount),
        dca_cadence: config.dcaCadence,
        buy_dip_threshold: parseFloat(config.buyDipThreshold)
      });
      setResult(response.data);
      toast.success('Backtest completed successfully');
    } catch (error) {
      console.error('Error running backtest:', error);
      toast.error('Failed to run backtest');
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="p-8" data-testid="backtest-lab-page">
      <div className="mb-8">
        <h1 className="text-4xl font-bold tracking-tight mb-2" data-testid="backtest-title">BACKTEST LAB</h1>
        <p className="text-muted-foreground">Simulate historical DCA performance with buy-the-dip strategy</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Configuration Panel */}
        <div className="glass-effect rounded-sm p-6" data-testid="backtest-config">
          <h2 className="text-xl font-bold mb-6">CONFIGURATION</h2>
          
          <div className="space-y-4">
            <div>
              <label className="text-sm text-muted-foreground mb-2 block">ASSET</label>
              <select
                value={config.symbol}
                onChange={(e) => setConfig({ ...config, symbol: e.target.value })}
                className="w-full p-3 glass-effect rounded text-sm font-data"
                data-testid="backtest-asset-select"
              >
                {assets.map(asset => (
                  <option key={asset.symbol} value={asset.symbol}>{asset.symbol} - {asset.name}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-sm text-muted-foreground mb-2 block">START DATE</label>
              <input
                type="date"
                value={config.startDate}
                onChange={(e) => setConfig({ ...config, startDate: e.target.value })}
                className="w-full p-3 glass-effect rounded text-sm font-data"
                data-testid="backtest-start-date"
              />
            </div>

            <div>
              <label className="text-sm text-muted-foreground mb-2 block">END DATE</label>
              <input
                type="date"
                value={config.endDate}
                onChange={(e) => setConfig({ ...config, endDate: e.target.value })}
                className="w-full p-3 glass-effect rounded text-sm font-data"
                data-testid="backtest-end-date"
              />
            </div>

            <div>
              <label className="text-sm text-muted-foreground mb-2 block">DCA AMOUNT (₹)</label>
              <input
                type="number"
                value={config.dcaAmount}
                onChange={(e) => setConfig({ ...config, dcaAmount: e.target.value })}
                className="w-full p-3 glass-effect rounded text-sm font-data"
                data-testid="backtest-dca-amount"
              />
            </div>

            <div>
              <label className="text-sm text-muted-foreground mb-2 block">DCA CADENCE</label>
              <select
                value={config.dcaCadence}
                onChange={(e) => setConfig({ ...config, dcaCadence: e.target.value })}
                className="w-full p-3 glass-effect rounded text-sm font-data"
                data-testid="backtest-cadence-select"
              >
                <option value="weekly">Weekly</option>
                <option value="monthly">Monthly</option>
              </select>
            </div>

            <div>
              <label className="text-sm text-muted-foreground mb-2 block">BUY DIP THRESHOLD (Score)</label>
              <input
                type="number"
                value={config.buyDipThreshold}
                onChange={(e) => setConfig({ ...config, buyDipThreshold: e.target.value })}
                className="w-full p-3 glass-effect rounded text-sm font-data"
                min="0"
                max="100"
                data-testid="backtest-dip-threshold"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Invest extra 50% when score exceeds this threshold
              </p>
            </div>

            <button
              onClick={runBacktest}
              disabled={running}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary text-primary-foreground rounded font-medium hover:bg-primary/90 transition disabled:opacity-50"
              data-testid="run-backtest-btn"
            >
              <Play className="w-5 h-5" />
              {running ? 'RUNNING...' : 'RUN BACKTEST'}
            </button>
          </div>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-6">
          {!result ? (
            <div className="glass-effect rounded-sm p-12 text-center" data-testid="backtest-empty">
              <TrendingUp className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-xl font-bold mb-2">NO RESULTS YET</h3>
              <p className="text-muted-foreground">
                Configure your backtest parameters and click "RUN BACKTEST" to see historical DCA performance.
              </p>
            </div>
          ) : (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-2 gap-4" data-testid="backtest-results">
                <div className="glass-effect rounded-sm p-6">
                  <div className="text-sm text-muted-foreground mb-1">TOTAL INVESTED</div>
                  <div className="text-3xl font-bold font-data">
                    {formatCurrency(result.total_invested, 'USD')}
                  </div>
                </div>

                <div className="glass-effect rounded-sm p-6">
                  <div className="text-sm text-muted-foreground mb-1">FINAL VALUE (USD)</div>
                  <div className="text-3xl font-bold font-data">
                    {formatCurrency(result.final_value_usd, 'USD')}
                  </div>
                </div>

                <div className="glass-effect rounded-sm p-6">
                  <div className="text-sm text-muted-foreground mb-1">TOTAL RETURN</div>
                  <div className={`text-3xl font-bold font-data ${
                    result.total_return_pct >= 0 ? 'text-chart-4' : 'text-destructive'
                  }`}>
                    {formatPercent(result.total_return_pct, 2)}
                  </div>
                </div>

                <div className="glass-effect rounded-sm p-6">
                  <div className="text-sm text-muted-foreground mb-1">ANNUALIZED RETURN</div>
                  <div className={`text-3xl font-bold font-data ${
                    result.annualized_return_pct >= 0 ? 'text-chart-4' : 'text-destructive'
                  }`}>
                    {formatPercent(result.annualized_return_pct, 2)}
                  </div>
                </div>
              </div>

              {/* Details */}
              <div className="glass-effect rounded-sm p-6">
                <h3 className="text-xl font-bold mb-4">INVESTMENT DETAILS</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">REGULAR DCA PURCHASES</div>
                    <div className="text-2xl font-bold font-data">{result.num_regular_dca}</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">DIP PURCHASES</div>
                    <div className="text-2xl font-bold font-data text-primary">{result.num_dip_buys}</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">TOTAL UNITS ACQUIRED</div>
                    <div className="text-2xl font-bold font-data">{formatNumber(result.total_units, 4)}</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">FINAL VALUE (INR)</div>
                    <div className="text-2xl font-bold font-data">{formatCurrency(result.final_value_inr, 'INR')}</div>
                  </div>
                </div>
              </div>

              {/* Insight */}
              <div className="glass-effect rounded-sm p-6 border-l-4 border-primary">
                <h3 className="font-bold mb-2">BACKTEST INSIGHT</h3>
                <p className="text-sm text-muted-foreground">
                  {result.num_dip_buys > 0 ? (
                    <>Your buy-the-dip strategy triggered {result.num_dip_buys} additional purchases when the DCA score exceeded {config.buyDipThreshold}. 
                    This systematic approach to buying dips resulted in {result.total_return_pct >= 0 ? 'positive' : 'negative'} returns over the period.</>                  ) : (
                    <>No dip-buying opportunities met your threshold during this period. Consider adjusting your threshold or testing different time periods.</>                  )}
                </p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default BacktestLab;
