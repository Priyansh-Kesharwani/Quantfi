import React, { useState, useEffect, useMemo } from 'react';
import { useWatchlist } from '../contexts/WatchlistContext';
import { PageShell, StatCard, MetricGrid } from '../components/shared';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import api from '../api';
import { formatCurrency, formatPercent, formatNumber } from '../utils';
import { toast } from 'sonner';
import {
  Play, TrendingUp, TrendingDown, BarChart3, Activity, Shield,
  Clock, Zap, DollarSign, Target, AlertTriangle, CheckCircle,
  XCircle, ArrowUpRight, ArrowDownRight, Layers
} from 'lucide-react';
import {
  ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, BarChart, Bar, Cell, PieChart, Pie
} from 'recharts';

/* ─── Strategy Template Presets (entry/max_positions sync with backend SIMULATION_TEMPLATES) ─── */
const TEMPLATES = {
  conservative: { label: 'CONSERVATIVE', desc: 'Tight stops, high entry bar, fewer positions', color: '#22C55E', entryThreshold: 80, maxPositions: 6 },
  balanced:     { label: 'BALANCED',     desc: 'Default mean-reversion parameters',         color: '#6366F1', entryThreshold: 70, maxPositions: 10 },
  aggressive:   { label: 'AGGRESSIVE',   desc: 'Loose stops, lower entry bar, more slots',  color: '#EF4444', entryThreshold: 60, maxPositions: 15 },
};

/* ─── Overview Tab ─── */
const OverviewTab = ({ result, config }) => {
  const assetPriceSymbols = useMemo(() => {
    const prices = result?.benchmarks?.asset_prices || {};
    return Object.keys(prices);
  }, [result]);

  const chartData = useMemo(() => {
    if (!result?.equity_curve?.length) return [];
    const strat = result.equity_curve;
    const bnh = result.benchmarks?.buy_and_hold?.equity_curve || [];
    const unif = result.benchmarks?.uniform_periodic?.equity_curve || [];
    const assetPrices = result.benchmarks?.asset_prices || {};
    const bnhMap = Object.fromEntries(bnh.map(p => [p.date, p.equity]));
    const unifMap = Object.fromEntries(unif.map(p => [p.date, p.equity]));
    const assetMaps = {};
    for (const [sym, curve] of Object.entries(assetPrices)) {
      assetMaps[sym] = Object.fromEntries(curve.map(p => [p.date, p.value]));
    }
    return strat.map(pt => {
      const row = {
        date: pt.date,
        strategy: pt.equity,
        buyHold: bnhMap[pt.date] || null,
        uniform: unifMap[pt.date] || null,
        cash: pt.cash,
        invested_pct: pt.invested_pct,
      };
      for (const sym of Object.keys(assetMaps)) {
        row[sym] = assetMaps[sym][pt.date] || null;
      }
      return row;
    });
  }, [result]);

  if (!result) return null;

  const bnh = result.benchmarks?.buy_and_hold || {};
  const unif = result.benchmarks?.uniform_periodic || {};

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={TrendingUp} label="Total Return" value={formatPercent(result.total_return_pct, 2)}
          trendPositive={result.total_return_pct >= 0} />
        <StatCard icon={BarChart3} label="CAGR" value={formatPercent(result.cagr_pct, 2)}
          trendPositive={result.cagr_pct >= 0} />
        <StatCard icon={Activity} label="Sharpe Ratio" value={result.sharpe_ratio?.toFixed(3)} />
        <StatCard icon={Shield} label="Max Drawdown" value={formatPercent(result.max_drawdown_pct, 2)} />
      </div>

      {/* Equity Curve */}
      {chartData.length > 0 && (
        <div className="glass-effect rounded-sm p-6">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5" /> PORTFOLIO vs BENCHMARKS
          </h3>
          <div style={{ width: '100%', height: 380 }}>
            <ResponsiveContainer>
              <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="stratGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6366F1" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#6366F1" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#888' }}
                  tickFormatter={d => d?.substring(0, 7)}
                  interval={Math.max(1, Math.floor(chartData.length / 8))} />
                <YAxis tick={{ fontSize: 10, fill: '#888' }}
                  tickFormatter={v => `$${(v / 1000).toFixed(0)}k`} />
                <Tooltip content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null;
                  return (
                    <div className="glass-effect rounded p-3 text-xs border border-white/10">
                      <p className="font-bold mb-1">{label}</p>
                      {payload.map((e, i) => (
                        <p key={i} style={{ color: e.color }}>
                          {e.name}: {formatCurrency(e.value, 'USD')}
                        </p>
                      ))}
                    </div>
                  );
                }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Area type="monotone" dataKey="strategy" name="Score-Guided" stroke="#6366F1"
                  fill="url(#stratGrad)" strokeWidth={2.5} />
                <Line type="monotone" dataKey="buyHold" name="Buy & Hold" stroke="#22C55E"
                  strokeWidth={1.5} strokeDasharray="5 3" dot={false} connectNulls />
                <Line type="monotone" dataKey="uniform" name="Uniform Monthly" stroke="#F59E0B"
                  strokeWidth={1.5} strokeDasharray="3 3" dot={false} connectNulls />
                {assetPriceSymbols.map((sym, i) => {
                  const colors = ['#EC4899', '#14B8A6', '#F97316', '#8B5CF6', '#06B6D4'];
                  return (
                    <Line key={sym} type="monotone" dataKey={sym} name={sym}
                      stroke={colors[i % colors.length]} strokeWidth={1.2}
                      strokeDasharray="2 4" dot={false} connectNulls />
                  );
                })}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Strategy Comparison Table */}
      <div className="glass-effect rounded-sm p-6">
        <h3 className="text-lg font-bold mb-4">STRATEGY COMPARISON</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left text-[10px] text-muted-foreground py-2 uppercase">Strategy</th>
                <th className="text-right text-[10px] text-muted-foreground py-2 uppercase">Total Return</th>
                <th className="text-right text-[10px] text-muted-foreground py-2 uppercase">CAGR</th>
                <th className="text-right text-[10px] text-muted-foreground py-2 uppercase">Sharpe</th>
                <th className="text-right text-[10px] text-muted-foreground py-2 uppercase">Max DD</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-white/[0.04] bg-primary/5">
                <td className="py-3 text-sm font-bold text-primary">Score-Guided</td>
                <td className="py-3 text-right font-data text-sm">{formatPercent(result.total_return_pct, 2)}</td>
                <td className="py-3 text-right font-data text-sm">{formatPercent(result.cagr_pct, 2)}</td>
                <td className="py-3 text-right font-data text-sm">{result.sharpe_ratio?.toFixed(3)}</td>
                <td className="py-3 text-right font-data text-sm text-red-400">{formatPercent(result.max_drawdown_pct, 2)}</td>
              </tr>
              {bnh.total_return_pct != null && (
                <tr className="border-b border-white/[0.04]">
                  <td className="py-3 text-sm font-bold">Buy & Hold</td>
                  <td className="py-3 text-right font-data text-sm">{formatPercent(bnh.total_return_pct, 2)}</td>
                  <td className="py-3 text-right font-data text-sm">{formatPercent(bnh.cagr_pct, 2)}</td>
                  <td className="py-3 text-right font-data text-sm">{bnh.sharpe_ratio?.toFixed(3)}</td>
                  <td className="py-3 text-right font-data text-sm text-red-400">{formatPercent(bnh.max_drawdown_pct, 2)}</td>
                </tr>
              )}
              {unif.total_return_pct != null && (
                <tr className="border-b border-white/[0.04]">
                  <td className="py-3 text-sm font-bold">Uniform Monthly</td>
                  <td className="py-3 text-right font-data text-sm">{formatPercent(unif.total_return_pct, 2)}</td>
                  <td className="py-3 text-right font-data text-sm">{formatPercent(unif.cagr_pct, 2)}</td>
                  <td className="py-3 text-right font-data text-sm">—</td>
                  <td className="py-3 text-right font-data text-sm">—</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Secondary Metrics */}
      <div className="glass-effect rounded-sm p-6">
        <h3 className="text-lg font-bold mb-4">DETAILED METRICS</h3>
        <MetricGrid items={[
          { label: 'SORTINO RATIO', value: result.sortino_ratio?.toFixed(3) },
          { label: 'CALMAR RATIO', value: result.calmar_ratio?.toFixed(3) },
          { label: 'WIN RATE', value: `${result.win_rate?.toFixed(1)}%` },
          { label: 'AVG HOLDING', value: `${result.avg_holding_days?.toFixed(0)} days` },
          { label: 'TIME IN MARKET', value: `${result.time_in_market_pct?.toFixed(0)}%` },
          { label: 'TOTAL TRADES', value: result.total_trades },
          { label: 'TOTAL COSTS', value: formatCurrency(result.total_costs, 'USD') },
          { label: 'COST DRAG', value: `${result.cost_drag_pct?.toFixed(2)}%`, color: '#EF4444' },
        ]} cols={4} />
      </div>
    </div>
  );
};

/* ─── Trades Tab ─── */
const TradesTab = ({ result }) => {
  if (!result?.trades?.length) {
    return (
      <div className="glass-effect rounded-sm p-12 text-center">
        <Target className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
        <h3 className="text-xl font-bold mb-2">NO TRADES</h3>
        <p className="text-muted-foreground text-sm">No entry signals were triggered during this period.</p>
      </div>
    );
  }

  const exitReasons = result.exit_reasons || {};
  const reasonColors = { stop: '#EF4444', score: '#22C55E', time: '#F59E0B', end_of_sim: '#6366F1' };

  return (
    <div className="space-y-6">
      {/* Exit Reason Summary */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {Object.entries(exitReasons).map(([reason, count]) => (
          <div key={reason} className="glass-effect rounded-sm p-4">
            <div className="text-xs text-muted-foreground uppercase mb-1">{reason} EXITS</div>
            <div className="text-2xl font-bold font-data" style={{ color: reasonColors[reason] || '#888' }}>
              {count}
            </div>
          </div>
        ))}
      </div>

      {/* Trade Log */}
      <div className="glass-effect rounded-sm p-6">
        <h3 className="text-lg font-bold mb-4">TRADE LOG</h3>
        <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
          <table className="w-full">
            <thead className="sticky top-0 bg-background/90 backdrop-blur">
              <tr className="border-b border-white/10">
                <th className="text-left text-[10px] text-muted-foreground py-2 uppercase">Date</th>
                <th className="text-left text-[10px] text-muted-foreground py-2 uppercase">Symbol</th>
                <th className="text-center text-[10px] text-muted-foreground py-2 uppercase">Side</th>
                <th className="text-right text-[10px] text-muted-foreground py-2 uppercase">Price</th>
                <th className="text-right text-[10px] text-muted-foreground py-2 uppercase">Notional</th>
                <th className="text-right text-[10px] text-muted-foreground py-2 uppercase">Score</th>
                <th className="text-right text-[10px] text-muted-foreground py-2 uppercase">P&L</th>
                <th className="text-right text-[10px] text-muted-foreground py-2 uppercase">Hold</th>
                <th className="text-center text-[10px] text-muted-foreground py-2 uppercase">Reason</th>
              </tr>
            </thead>
            <tbody>
              {result.trades.map((t, i) => (
                <tr key={i} className="border-b border-white/[0.04] hover:bg-white/[0.02]">
                  <td className="py-2 text-xs font-data">{t.date}</td>
                  <td className="py-2 text-xs font-bold">{t.symbol}</td>
                  <td className="py-2 text-center">
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                      t.side === 'ENTRY' ? 'bg-emerald-400/10 text-emerald-400' : 'bg-red-400/10 text-red-400'
                    }`}>{t.side}</span>
                  </td>
                  <td className="py-2 text-right text-xs font-data">{formatCurrency(t.price, 'USD')}</td>
                  <td className="py-2 text-right text-xs font-data">{formatCurrency(t.notional, 'USD')}</td>
                  <td className="py-2 text-right text-xs font-data">{t.score?.toFixed(0)}</td>
                  <td className={`py-2 text-right text-xs font-data font-bold ${
                    t.pnl > 0 ? 'text-emerald-400' : t.pnl < 0 ? 'text-red-400' : ''
                  }`}>
                    {t.side === 'EXIT' ? formatCurrency(t.pnl, 'USD') : '—'}
                  </td>
                  <td className="py-2 text-right text-xs font-data">
                    {t.side === 'EXIT' ? `${t.holding_days}d` : '—'}
                  </td>
                  <td className="py-2 text-center">
                    {t.exit_reason && (
                      <span className="text-[9px] font-bold px-1.5 py-0.5 rounded"
                        style={{ color: reasonColors[t.exit_reason] || '#888',
                                 backgroundColor: `${reasonColors[t.exit_reason] || '#888'}20` }}>
                        {t.exit_reason.toUpperCase()}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

/* ─── Asset Breakdown Tab ─── */
const AssetBreakdownTab = ({ result }) => {
  const breakdown = result?.asset_breakdown || {};
  const entries = Object.entries(breakdown);

  if (entries.length === 0) {
    return (
      <div className="glass-effect rounded-sm p-12 text-center">
        <Layers className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
        <h3 className="text-xl font-bold mb-2">NO ASSET DATA</h3>
        <p className="text-muted-foreground text-sm">Run a simulation to see per-asset breakdown.</p>
      </div>
    );
  }

  const barData = entries.map(([sym, data]) => ({
    name: sym,
    pnl: data.total_pnl,
    trades: data.trades,
    fill: data.total_pnl >= 0 ? '#22C55E' : '#EF4444',
  })).sort((a, b) => b.pnl - a.pnl);

  return (
    <div className="space-y-6">
      {/* PnL by Asset */}
      {barData.length > 0 && (
        <div className="glass-effect rounded-sm p-6">
          <h3 className="text-lg font-bold mb-4">P&L BY ASSET</h3>
          <div style={{ width: '100%', height: Math.max(200, barData.length * 40) }}>
            <ResponsiveContainer>
              <BarChart data={barData} layout="vertical" margin={{ top: 5, right: 30, left: 60, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis type="number" tick={{ fontSize: 10, fill: '#888' }}
                  tickFormatter={v => `$${v.toFixed(0)}`} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: '#888' }} />
                <Tooltip content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  return (
                    <div className="glass-effect rounded p-2 text-xs border border-white/10">
                      <p className="font-bold">{payload[0].payload.name}</p>
                      <p style={{ color: payload[0].payload.fill }}>
                        P&L: {formatCurrency(payload[0].value, 'USD')}
                      </p>
                      <p className="text-muted-foreground">{payload[0].payload.trades} trades</p>
                    </div>
                  );
                }} />
                <Bar dataKey="pnl" radius={[0, 4, 4, 0]} barSize={24}>
                  {barData.map((entry, i) => <Cell key={i} fill={entry.fill} fillOpacity={0.7} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Asset Detail Table */}
      <div className="glass-effect rounded-sm p-6">
        <h3 className="text-lg font-bold mb-4">PER-ASSET DETAILS</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left text-[10px] text-muted-foreground py-2 uppercase">Asset</th>
                <th className="text-right text-[10px] text-muted-foreground py-2 uppercase">Trades</th>
                <th className="text-right text-[10px] text-muted-foreground py-2 uppercase">Total P&L</th>
                <th className="text-right text-[10px] text-muted-foreground py-2 uppercase">Win Rate</th>
                <th className="text-right text-[10px] text-muted-foreground py-2 uppercase">Avg Hold</th>
                <th className="text-right text-[10px] text-muted-foreground py-2 uppercase">Costs</th>
              </tr>
            </thead>
            <tbody>
              {entries.map(([sym, data]) => (
                <tr key={sym} className="border-b border-white/[0.04]">
                  <td className="py-3 text-sm font-bold">{sym}</td>
                  <td className="py-3 text-right font-data text-sm">{data.trades}</td>
                  <td className={`py-3 text-right font-data text-sm font-bold ${
                    data.total_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'
                  }`}>{formatCurrency(data.total_pnl, 'USD')}</td>
                  <td className="py-3 text-right font-data text-sm">{data.win_rate?.toFixed(1)}%</td>
                  <td className="py-3 text-right font-data text-sm">{data.avg_holding_days?.toFixed(0)}d</td>
                  <td className="py-3 text-right font-data text-sm text-muted-foreground">
                    {formatCurrency(data.total_costs, 'USD')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

/* ─── Cost Analysis Tab ─── */
const CostTab = ({ result }) => {
  if (!result) return null;
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={DollarSign} label="Total Costs" value={formatCurrency(result.total_costs, 'USD')} />
        <StatCard icon={Activity} label="Cost Drag" value={`${result.cost_drag_pct?.toFixed(2)}%`} />
        <StatCard icon={TrendingUp} label="Gross Return" value={formatPercent(result.total_return_pct + result.cost_drag_pct, 2)} />
        <StatCard icon={Shield} label="Net Return" value={formatPercent(result.total_return_pct, 2)} />
      </div>

      <div className="glass-effect rounded-sm p-6 border-l-4 border-primary">
        <h3 className="font-bold mb-2">COST MODEL</h3>
        <p className="text-sm text-muted-foreground">
          Transaction costs are modelled per-asset-class: Indian equities ~40 bps round-trip + ₹20 flat,
          US equities from India ~140 bps (forex spread dominated), commodities ~30 bps, crypto ~80 bps.
          Slippage is applied at {result.config?.slippage_bps || 5} bps on execution.
          All costs are deducted from capital at the time of trade.
        </p>
      </div>

      {/* Per-trade cost distribution */}
      {result.trades?.length > 0 && (
        <div className="glass-effect rounded-sm p-6">
          <h3 className="text-lg font-bold mb-4">COST PER TRADE</h3>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-xs text-muted-foreground mb-1">AVG COST / TRADE</div>
              <div className="text-xl font-bold font-data">
                {formatCurrency(result.total_costs / Math.max(result.trades.length, 1), 'USD')}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground mb-1">TOTAL ROUND-TRIPS</div>
              <div className="text-xl font-bold font-data">{result.total_trades}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground mb-1">COST AS % OF TRADED NOTIONAL</div>
              <div className="text-xl font-bold font-data">
                {result.trades.length > 0 ? (
                  (result.total_costs / result.trades.reduce((s, t) => s + t.notional, 0) * 100).toFixed(3)
                ) : '0'}%
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

/* ════════════════════════════════════════════════════════════════
   MAIN PORTFOLIO SIMULATION PAGE
   ════════════════════════════════════════════════════════════════ */
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
      entryThreshold: t.entryThreshold,
      maxPositions: t.maxPositions,
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
          Mean-reversion backtesting with dynamic entry/exit signals — real market data
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* ─── Configuration Panel ─── */}
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
                    onClick={() => toggleAsset(asset.symbol)}
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
                    onClick={() => handleTemplateChange(key)}
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
                  onChange={e => updateConfig('startDate', e.target.value)}
                  className="w-full p-2 glass-effect rounded text-xs font-data" />
              </div>
              <div>
                <label className="text-[10px] text-muted-foreground mb-1 block">END DATE</label>
                <input type="date" value={config.endDate}
                  onChange={e => updateConfig('endDate', e.target.value)}
                  className="w-full p-2 glass-effect rounded text-xs font-data" />
              </div>
            </div>
            <div className="flex gap-2">
              {[3, 5, 10].map(y => (
                <button key={y} onClick={() => {
                  const d = new Date();
                  d.setFullYear(d.getFullYear() - y);
                  updateConfig('startDate', d.toISOString().split('T')[0]);
                }} className="px-2 py-1 text-xs glass-effect rounded hover:bg-white/10 transition">
                  {y}Y
                </button>
              ))}
            </div>

            {/* Capital */}
            <div>
              <label className="text-[10px] text-muted-foreground mb-1 block">INITIAL CAPITAL ($)</label>
              <input type="number" value={config.initialCapital}
                onChange={e => updateConfig('initialCapital', e.target.value)}
                className="w-full p-2 glass-effect rounded text-xs font-data" />
            </div>

            {/* Entry Threshold */}
            <div>
              <label className="text-[10px] text-muted-foreground mb-1 block">
                ENTRY SCORE THRESHOLD ({config.entryThreshold})
              </label>
              <input type="range" min="40" max="90" step="5"
                value={config.entryThreshold}
                onChange={e => updateConfig('entryThreshold', e.target.value)}
                className="w-full" />
              <p className="text-[10px] text-muted-foreground">
                Enter when composite score ≥ {config.entryThreshold}
              </p>
            </div>

            {/* Max Positions */}
            <div>
              <label className="text-[10px] text-muted-foreground mb-1 block">MAX POSITIONS</label>
              <input type="number" value={config.maxPositions} min="1" max="20"
                onChange={e => updateConfig('maxPositions', e.target.value)}
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

            {/* Run Button */}
            <button onClick={runSimulation} disabled={running}
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

        {/* ─── Results Panel ─── */}
        <div className="lg:col-span-2">
          {!result ? (
            <div className="glass-effect rounded-sm p-12 text-center" data-testid="sim-empty">
              <TrendingUp className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-xl font-bold mb-2">NO RESULTS YET</h3>
              <p className="text-muted-foreground text-sm">
                Select assets, configure your strategy, and click "RUN SIMULATION".
              </p>
              <p className="text-xs text-muted-foreground mt-2">
                Mean-reversion strategy: buys dips (high score), exits via trailing stop / score normalization / time decay
              </p>
            </div>
          ) : (
            <div className="w-full">
            {resultStale && (
              <div className="flex items-center gap-2 px-4 py-2.5 mb-4 rounded border border-yellow-500/30 bg-yellow-500/10 text-yellow-400 text-xs">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>Config changed since last run — click <strong>RUN SIMULATION</strong> to update results</span>
              </div>
            )}
            <Tabs defaultValue="overview" className="w-full">
              <TabsList className="glass-effect mb-6 w-full justify-start">
                <TabsTrigger value="overview" className="font-bold text-xs">OVERVIEW</TabsTrigger>
                <TabsTrigger value="trades" className="font-bold text-xs">TRADES</TabsTrigger>
                <TabsTrigger value="assets" className="font-bold text-xs">ASSETS</TabsTrigger>
                <TabsTrigger value="costs" className="font-bold text-xs">COSTS</TabsTrigger>
              </TabsList>
              <TabsContent value="overview">
                <OverviewTab result={result} config={config} />
              </TabsContent>
              <TabsContent value="trades">
                <TradesTab result={result} />
              </TabsContent>
              <TabsContent value="assets">
                <AssetBreakdownTab result={result} />
              </TabsContent>
              <TabsContent value="costs">
                <CostTab result={result} />
              </TabsContent>
            </Tabs>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PortfolioSim;
