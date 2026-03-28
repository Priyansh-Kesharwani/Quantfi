import React, { useMemo } from 'react';
import { StatCard, MetricGrid } from '@/components/shared';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { formatCurrency, formatPercent } from '@/utils';
import {
  TrendingUp, BarChart3, Activity, Shield, DollarSign,
  Target, AlertTriangle, Layers,
} from 'lucide-react';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Cell,
} from 'recharts';
import EquityCurve from './EquityCurve';

/* ─── Overview Tab ─── */
const OverviewTab = ({ result, config }) => {
  if (!result) return null;

  const bnh = result.benchmarks?.buy_and_hold || {};
  const unif = result.benchmarks?.uniform_periodic || {};

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={TrendingUp} label="Total Return" value={formatPercent(result.total_return_pct, 2)}
          trendPositive={result.total_return_pct >= 0} />
        <StatCard icon={BarChart3} label="CAGR" value={formatPercent(result.cagr_pct, 2)}
          trendPositive={result.cagr_pct >= 0} />
        <StatCard icon={Activity} label="Sharpe Ratio" value={result.sharpe_ratio?.toFixed(3)} />
        <StatCard icon={Shield} label="Max Drawdown" value={formatPercent(result.max_drawdown_pct, 2)} />
      </div>

      <EquityCurve result={result} />

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

  const barData = useMemo(() => {
    return entries.map(([sym, data]) => ({
      name: sym,
      pnl: data.total_pnl,
      trades: data.trades,
      fill: data.total_pnl >= 0 ? '#22C55E' : '#EF4444',
    })).sort((a, b) => b.pnl - a.pnl);
  }, [entries]);

  if (entries.length === 0) {
    return (
      <div className="glass-effect rounded-sm p-12 text-center">
        <Layers className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
        <h3 className="text-xl font-bold mb-2">NO ASSET DATA</h3>
        <p className="text-muted-foreground text-sm">Run a simulation to see per-asset breakdown.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
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

/* ─── Main Results Panel ─── */
const ResultsPanel = ({ result, config, resultStale }) => {
  if (!result) {
    return (
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
    );
  }

  return (
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
  );
};

export default ResultsPanel;
