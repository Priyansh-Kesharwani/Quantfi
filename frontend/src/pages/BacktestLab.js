import React, { useState, useEffect, useMemo } from 'react';
import api from '../api';
import { formatCurrency, formatPercent, formatNumber } from '../utils';
import { Play, TrendingUp, TrendingDown, DollarSign, Calendar, Database, BarChart3, AlertTriangle } from 'lucide-react';
import { toast } from 'sonner';
import {
  ResponsiveContainer, ComposedChart, Area, Line, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend
} from 'recharts';

const BacktestLab = ({ refreshKey = 0 }) => {
  const [assets, setAssets] = useState([]);
  const [config, setConfig] = useState({
    symbol: '',
    startDate: '1991-01-01',  // ~35 years of history by default
    endDate: new Date().toISOString().split('T')[0],
    dcaAmount: 5000,
    dcaCadence: 'monthly',
    buyDipThreshold: 60
  });
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchAssets();
  }, [refreshKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchAssets = async () => {
    try {
      const response = await api.getAssets();
      setAssets(response.data);
      if (response.data.length > 0) {
        setConfig(prev => ({ ...prev, symbol: response.data[0].symbol }));
      }
    } catch (err) {
      console.error('Error fetching assets:', err);
    }
  };

  const runBacktest = async () => {
    if (!config.symbol) {
      toast.error('Please select an asset');
      return;
    }

    setRunning(true);
    setError(null);
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
      toast.success('Backtest completed — real market data');
    } catch (err) {
      console.error('Error running backtest:', err);
      const detail = err?.response?.data?.detail || 'Failed to run backtest';
      setError(detail);
      toast.error(detail);
    } finally {
      setRunning(false);
    }
  };

  // Prepare chart data with dip-buy markers
  const chartData = useMemo(() => {
    if (!result?.equity_curve?.length) return [];
    return result.equity_curve.map(pt => ({
      date: pt.date,
      portfolio: pt.portfolio_value,
      invested: pt.total_invested,
      price: pt.price,
      score: pt.score,
      // dipBuy field: defined only where dip buying occurred (renders as dots)
      dipBuy: pt.is_dip_buy ? pt.portfolio_value : undefined,
    }));
  }, [result]);

  const dipBuyCount = useMemo(
    () => chartData.filter(pt => pt.dipBuy !== undefined).length,
    [chartData]
  );

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    return (
      <div className="glass-effect rounded p-3 text-xs border border-white/10">
        <p className="font-bold mb-1">{label}</p>
        {payload.map((entry, i) => (
          <p key={i} style={{ color: entry.color }}>
            {entry.name}: {entry.name === 'Score' ? entry.value?.toFixed(1) : formatCurrency(entry.value, 'USD')}
          </p>
        ))}
      </div>
    );
  };

  // Quick presets for date range
  const setDatePreset = (yearsBack) => {
    const d = new Date();
    d.setFullYear(d.getFullYear() - yearsBack);
    setConfig(prev => ({ ...prev, startDate: d.toISOString().split('T')[0] }));
  };

  return (
    <div className="p-8" data-testid="backtest-lab-page">
      <div className="mb-8">
        <h1 className="text-4xl font-bold tracking-tight mb-2" data-testid="backtest-title">BACKTEST LAB</h1>
        <p className="text-muted-foreground">Simulate historical DCA performance with buy-the-dip strategy — powered by real market data</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* ─── Configuration Panel ─── */}
        <div className="glass-effect rounded-sm p-6" data-testid="backtest-config">
          <h2 className="text-xl font-bold mb-6">CONFIGURATION</h2>

          <div className="space-y-4">
            {/* Asset */}
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

            {/* Date range */}
            <div>
              <label className="text-sm text-muted-foreground mb-2 block">START DATE</label>
              <input
                type="date"
                value={config.startDate}
                onChange={(e) => setConfig({ ...config, startDate: e.target.value })}
                className="w-full p-3 glass-effect rounded text-sm font-data"
                data-testid="backtest-start-date"
              />
              <div className="flex gap-2 mt-2">
                {[5, 10, 20, 35].map(y => (
                  <button
                    key={y}
                    onClick={() => setDatePreset(y)}
                    className="px-2 py-1 text-xs glass-effect rounded hover:bg-white/10 transition"
                  >
                    {y}Y
                  </button>
                ))}
                <button
                  onClick={() => setConfig(prev => ({ ...prev, startDate: '1990-01-01' }))}
                  className="px-2 py-1 text-xs glass-effect rounded hover:bg-white/10 transition"
                >
                  MAX
                </button>
              </div>
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

            {/* DCA Amount */}
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

            {/* Cadence */}
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

            {/* Threshold */}
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
                Invest extra 50% when composite score ≥ this threshold
              </p>
            </div>

            {/* Run button */}
            <button
              onClick={runBacktest}
              disabled={running}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary text-primary-foreground rounded font-medium hover:bg-primary/90 transition disabled:opacity-50"
              data-testid="run-backtest-btn"
            >
              <Play className="w-5 h-5" />
              {running ? 'COMPUTING SCORES & SIMULATING…' : 'RUN BACKTEST'}
            </button>

            {error && (
              <div className="flex items-start gap-2 p-3 rounded bg-destructive/10 text-destructive text-xs">
                <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>
        </div>

        {/* ─── Results Panel ─── */}
        <div className="lg:col-span-2 space-y-6">
          {!result ? (
            <div className="glass-effect rounded-sm p-12 text-center" data-testid="backtest-empty">
              <TrendingUp className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-xl font-bold mb-2">NO RESULTS YET</h3>
              <p className="text-muted-foreground">
                Configure your backtest parameters and click "RUN BACKTEST" to see historical DCA performance.
              </p>
              <p className="text-xs text-muted-foreground mt-2">
                Uses real market data from Yahoo Finance — no synthetic data
              </p>
            </div>
          ) : (
            <>
              {/* Data source badge */}
              <div className="flex items-center gap-3 flex-wrap">
                <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-chart-4/20 text-chart-4">
                  <Database className="w-3 h-3" />
                  REAL DATA — {result.data_source || 'yfinance'}
                </span>
                <span className="text-xs text-muted-foreground">
                  {result.data_points?.toLocaleString()} data points
                  {result.data_start && result.data_end && ` · ${result.data_start} → ${result.data_end}`}
                </span>
              </div>

              {/* Summary Cards — 2×2 */}
              <div className="grid grid-cols-2 gap-4" data-testid="backtest-results">
                <div className="glass-effect rounded-sm p-6">
                  <div className="text-sm text-muted-foreground mb-1">TOTAL INVESTED</div>
                  <div className="text-3xl font-bold font-data">
                    {formatCurrency(result.total_invested, 'USD')}
                  </div>
                </div>

                <div className="glass-effect rounded-sm p-6">
                  <div className="text-sm text-muted-foreground mb-1">FINAL VALUE (USD)</div>
                  <div className={`text-3xl font-bold font-data ${
                    result.final_value_usd >= result.total_invested ? 'text-chart-4' : 'text-destructive'
                  }`}>
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

              {/* Equity Curve Chart */}
              {chartData.length > 0 && (
                <div className="glass-effect rounded-sm p-6">
                  <h3 className="text-xl font-bold mb-1 flex items-center gap-2">
                    <BarChart3 className="w-5 h-5" />
                    EQUITY CURVE
                  </h3>
                  {dipBuyCount > 0 && (
                    <p className="text-xs text-muted-foreground mb-4">
                      <span className="inline-block w-2.5 h-2.5 rounded-full bg-amber-400 mr-1 align-middle" />
                      {dipBuyCount} dip-buy event{dipBuyCount > 1 ? 's' : ''} marked on chart (score ≥ {config.buyDipThreshold})
                    </p>
                  )}
                  <div style={{ width: '100%', height: 380 }}>
                    <ResponsiveContainer>
                      <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                        <defs>
                          <linearGradient id="portfolioGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#22C55E" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#22C55E" stopOpacity={0} />
                          </linearGradient>
                          <linearGradient id="investedGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#6366F1" stopOpacity={0.2} />
                            <stop offset="95%" stopColor="#6366F1" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                        <XAxis
                          dataKey="date"
                          tick={{ fontSize: 10, fill: '#888' }}
                          tickFormatter={(d) => d?.substring(0, 7)}
                          interval={Math.max(1, Math.floor(chartData.length / 8))}
                        />
                        <YAxis
                          tick={{ fontSize: 10, fill: '#888' }}
                          tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                        />
                        <Tooltip content={<CustomTooltip />} />
                        <Legend wrapperStyle={{ fontSize: 12 }} />
                        <Area
                          type="monotone"
                          dataKey="portfolio"
                          name="Portfolio Value"
                          stroke="#22C55E"
                          fill="url(#portfolioGradient)"
                          strokeWidth={2}
                        />
                        <Area
                          type="monotone"
                          dataKey="invested"
                          name="Total Invested"
                          stroke="#6366F1"
                          fill="url(#investedGradient)"
                          strokeWidth={1.5}
                          strokeDasharray="4 2"
                        />
                        {/* Dip-buy markers — golden diamonds on the portfolio line */}
                        <Line
                          type="monotone"
                          dataKey="dipBuy"
                          name="Dip Buy"
                          stroke="none"
                          dot={{ fill: '#FBBF24', r: 4, stroke: '#F59E0B', strokeWidth: 2 }}
                          activeDot={{ r: 7, fill: '#FBBF24', stroke: '#fff', strokeWidth: 2 }}
                          legendType="diamond"
                          connectNulls={false}
                          isAnimationActive={false}
                        />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Detailed Metrics */}
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

                {/* Extra metrics row */}
                <div className="grid grid-cols-2 md:grid-cols-3 gap-6 mt-6 pt-6 border-t border-white/10">
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">AVG COST BASIS</div>
                    <div className="text-xl font-bold font-data">
                      {result.avg_cost_basis ? formatCurrency(result.avg_cost_basis, 'USD') : '—'}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">MAX DRAWDOWN</div>
                    <div className="text-xl font-bold font-data text-destructive">
                      {result.max_drawdown_pct != null ? `${result.max_drawdown_pct.toFixed(2)}%` : '—'}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">DIP BUY RATIO</div>
                    <div className="text-xl font-bold font-data text-primary">
                      {result.num_regular_dca > 0
                        ? `${((result.num_dip_buys / result.num_regular_dca) * 100).toFixed(1)}%`
                        : '—'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Insight */}
              <div className="glass-effect rounded-sm p-6 border-l-4 border-primary">
                <h3 className="font-bold mb-2">BACKTEST INSIGHT</h3>
                <p className="text-sm text-muted-foreground">
                  {result.num_dip_buys > 0 ? (
                    <>
                      Your buy-the-dip strategy triggered <strong className="text-primary">{result.num_dip_buys}</strong> additional
                      purchases (out of {result.num_regular_dca} DCA events) when the composite score exceeded {config.buyDipThreshold}.
                      {result.avg_cost_basis > 0 && (
                        <> Average cost basis: <strong>{formatCurrency(result.avg_cost_basis, 'USD')}</strong>.</>
                      )}
                      {result.max_drawdown_pct < -10 && (
                        <> Portfolio experienced a max drawdown of <strong className="text-destructive">{result.max_drawdown_pct.toFixed(1)}%</strong>,
                        but the DCA approach helps smooth entry prices.</>
                      )}
                      {' '}This resulted in a {result.total_return_pct >= 0 ? 'positive' : 'negative'} total return
                      of <strong className={result.total_return_pct >= 0 ? 'text-chart-4' : 'text-destructive'}>
                        {formatPercent(result.total_return_pct, 2)}
                      </strong> ({formatPercent(result.annualized_return_pct, 2)} annualised).
                    </>
                  ) : (
                    <>
                      No dip-buying opportunities met your threshold of {config.buyDipThreshold} during this period.
                      Consider lowering the threshold or testing a different date range.
                      All {result.num_regular_dca} DCA purchases were at the base investment amount.
                    </>
                  )}
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
