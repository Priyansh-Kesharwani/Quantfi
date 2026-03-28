import React, { useState, useMemo, useCallback } from 'react';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import OHLCVChart from '@/components/crypto/OHLCVChart';
import RegimePanel from '@/components/crypto/RegimePanel';
import {
  Filter,
  ArrowUpDown,
  ArrowUpRight,
  ArrowDownRight,
  Download,
  Grid3X3,
} from 'lucide-react';

const MetricCard = ({ label, value, suffix = '', positive }) => (
  <div className="glass-effect p-3 rounded-lg">
    <p className="text-xs text-muted-foreground uppercase tracking-wider">{label}</p>
    <p
      className={`text-lg font-bold mt-1 ${
        positive === true
          ? 'text-green-400'
          : positive === false
            ? 'text-red-400'
            : 'text-foreground'
      }`}
    >
      {value}
      {suffix}
    </p>
  </div>
);

const SIDE_LABELS = {
  long_entry: 'Long Entry',
  long_exit: 'Long Exit',
  short_entry: 'Short Entry',
  short_exit: 'Short Exit',
  grid_buy: 'Grid Buy',
  grid_sell: 'Grid Sell',
};

const SIDE_COLORS = {
  long_entry: 'text-green-400',
  long_exit: 'text-green-600',
  short_entry: 'text-red-400',
  short_exit: 'text-red-600',
  grid_buy: 'text-blue-400',
  grid_sell: 'text-blue-600',
};

function TradeTable({ trades = [] }) {
  const [sortKey, setSortKey] = useState('bar_idx');
  const [sortAsc, setSortAsc] = useState(true);
  const [sideFilter, setSideFilter] = useState('all');
  const [pnlFilter, setPnlFilter] = useState('all');
  const [page, setPage] = useState(0);
  const pageSize = 20;

  const filtered = useMemo(() => {
    let t = [...trades];
    if (sideFilter === 'long') t = t.filter((x) => x.side.startsWith('long'));
    else if (sideFilter === 'short') t = t.filter((x) => x.side.startsWith('short'));
    else if (sideFilter === 'grid') t = t.filter((x) => x.side.startsWith('grid'));
    if (pnlFilter === 'winners') t = t.filter((x) => x.pnl > 0);
    else if (pnlFilter === 'losers') t = t.filter((x) => x.pnl < 0);
    return t;
  }, [trades, sideFilter, pnlFilter]);

  const sorted = useMemo(() => {
    const s = [...filtered];
    s.sort((a, b) => {
      let av = a[sortKey],
        bv = b[sortKey];
      if (typeof av === 'string') av = av.toLowerCase();
      if (typeof bv === 'string') bv = bv.toLowerCase();
      if (av < bv) return sortAsc ? -1 : 1;
      if (av > bv) return sortAsc ? 1 : -1;
      return 0;
    });
    return s;
  }, [filtered, sortKey, sortAsc]);

  const pageCount = Math.ceil(sorted.length / pageSize);
  const pageRows = sorted.slice(page * pageSize, (page + 1) * pageSize);

  const toggleSort = (key) => {
    if (sortKey === key) setSortAsc(!sortAsc);
    else {
      setSortKey(key);
      setSortAsc(true);
    }
  };

  const totals = useMemo(() => {
    const pnls = filtered.filter((t) => t.pnl !== 0).map((t) => t.pnl);
    const wins = pnls.filter((p) => p > 0);
    return {
      totalPnl: pnls.reduce((a, b) => a + b, 0),
      totalFees: filtered.reduce((a, t) => a + (t.fee || 0), 0),
      winRate: pnls.length > 0 ? (wins.length / pnls.length) * 100 : 0,
      count: filtered.length,
    };
  }, [filtered]);

  const exportCSV = useCallback(() => {
    const cols = [
      'timestamp', 'side', 'price', 'units', 'notional', 'pnl',
      'fee', 'funding_paid', 'exit_reason', 'leverage', 'regime', 'bar_idx',
    ];
    const header = cols.join(',');
    const rows = sorted.map((t) =>
      cols
        .map((c) => {
          const v = t[c];
          if (typeof v === 'string' && v.includes(',')) return `"${v}"`;
          if (typeof v === 'number') return Number(v.toFixed(4));
          return v ?? '';
        })
        .join(',')
    );
    const csv = [header, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trades_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [sorted]);

  const SortHeader = ({ label, field, className = '' }) => (
    <th
      className={`px-2 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider cursor-pointer select-none hover:text-foreground transition-colors ${className}`}
      onClick={() => toggleSort(field)}
    >
      <span className="flex items-center gap-1">
        {label}
        {sortKey === field && <ArrowUpDown className="h-3 w-3" />}
      </span>
    </th>
  );

  if (trades.length === 0) return null;

  return (
    <Card className="glass-effect">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-3">
            <CardTitle className="text-sm">TRADE LOG</CardTitle>
            <span className="text-xs font-mono text-muted-foreground">
              {totals.count} trades · PnL:{' '}
              <span className={totals.totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                ${totals.totalPnl.toFixed(2)}
              </span>
              {' · '}Win: {totals.winRate.toFixed(0)}%
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <Filter className="h-3 w-3 text-muted-foreground" />
              <Select
                value={sideFilter}
                onValueChange={(v) => {
                  setSideFilter(v);
                  setPage(0);
                }}
              >
                <SelectTrigger className="h-7 text-xs w-[100px] glass-effect">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Sides</SelectItem>
                  <SelectItem value="long">Long</SelectItem>
                  <SelectItem value="short">Short</SelectItem>
                  <SelectItem value="grid">Grid</SelectItem>
                </SelectContent>
              </Select>
              <Select
                value={pnlFilter}
                onValueChange={(v) => {
                  setPnlFilter(v);
                  setPage(0);
                }}
              >
                <SelectTrigger className="h-7 text-xs w-[100px] glass-effect">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All PnL</SelectItem>
                  <SelectItem value="winners">Winners</SelectItem>
                  <SelectItem value="losers">Losers</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="h-7 gap-1 text-xs"
              onClick={exportCSV}
            >
              <Download className="h-3 w-3" /> CSV
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border">
                <SortHeader label="#" field="bar_idx" className="w-12" />
                <SortHeader label="Time" field="timestamp" />
                <SortHeader label="Side" field="side" />
                <SortHeader label="Price" field="price" />
                <SortHeader label="Size" field="notional" />
                <SortHeader label="PnL" field="pnl" />
                <SortHeader label="Fee" field="fee" />
                <SortHeader label="Regime" field="regime" />
                <SortHeader label="Exit Reason" field="exit_reason" />
              </tr>
            </thead>
            <tbody>
              {pageRows.map((t, i) => (
                <tr
                  key={i}
                  className="border-b border-border/40 hover:bg-muted/30 transition-colors"
                >
                  <td className="px-2 py-1.5 font-mono text-muted-foreground">
                    {t.bar_idx}
                  </td>
                  <td className="px-2 py-1.5 font-mono">
                    {t.timestamp
                      ? new Date(t.timestamp).toLocaleDateString('en-US', {
                          month: 'short',
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit',
                        })
                      : '—'}
                  </td>
                  <td className="px-2 py-1.5">
                    <span
                      className={`flex items-center gap-1 font-medium ${SIDE_COLORS[t.side] || ''}`}
                    >
                      {t.side?.startsWith('long') || t.side === 'grid_buy' ? (
                        <ArrowUpRight className="h-3 w-3" />
                      ) : t.side?.startsWith('grid') ? (
                        <Grid3X3 className="h-3 w-3" />
                      ) : (
                        <ArrowDownRight className="h-3 w-3" />
                      )}
                      {SIDE_LABELS[t.side] || t.side}
                    </span>
                  </td>
                  <td className="px-2 py-1.5 font-mono">
                    $
                    {t.price >= 1
                      ? t.price.toLocaleString(undefined, { maximumFractionDigits: 2 })
                      : t.price?.toPrecision(4)}
                  </td>
                  <td className="px-2 py-1.5 font-mono">${t.notional?.toFixed(0)}</td>
                  <td
                    className={`px-2 py-1.5 font-mono font-medium ${
                      t.pnl > 0
                        ? 'text-green-400'
                        : t.pnl < 0
                          ? 'text-red-400'
                          : 'text-muted-foreground'
                    }`}
                  >
                    {t.pnl === 0 ? '—' : `${t.pnl > 0 ? '+' : ''}$${t.pnl.toFixed(2)}`}
                  </td>
                  <td className="px-2 py-1.5 font-mono text-muted-foreground">
                    ${t.fee?.toFixed(2)}
                  </td>
                  <td className="px-2 py-1.5">
                    <span
                      className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${
                        t.regime === 'TRENDING'
                          ? 'bg-blue-500/20 text-blue-400'
                          : t.regime === 'STRESS'
                            ? 'bg-red-500/20 text-red-400'
                            : 'bg-yellow-500/20 text-yellow-400'
                      }`}
                    >
                      {t.regime}
                    </span>
                  </td>
                  <td className="px-2 py-1.5 text-muted-foreground">{t.exit_reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {pageCount > 1 && (
          <div className="flex items-center justify-between mt-3 pt-2 border-t border-border">
            <span className="text-xs text-muted-foreground">
              Page {page + 1} of {pageCount} ({sorted.length} trades)
            </span>
            <div className="flex gap-1">
              <Button
                variant="outline"
                size="sm"
                className="h-6 px-2 text-xs"
                disabled={page === 0}
                onClick={() => setPage((p) => p - 1)}
              >
                Prev
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-6 px-2 text-xs"
                disabled={page >= pageCount - 1}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function BacktestPanel({ result, initialCapital }) {
  if (!result) return null;

  return (
    <div className="space-y-6">
      {result.data_source && (
        <div className="flex items-center gap-2 text-xs">
          <span className="px-2 py-0.5 rounded font-mono bg-green-500/20 text-green-400">
            LIVE DATA
          </span>
          {result.n_bars_actual > 0 && (
            <span className="text-muted-foreground">
              {result.n_bars_actual.toLocaleString()} bars
            </span>
          )}
          {result.start_date && result.end_date && (
            <span className="text-muted-foreground">
              {result.start_date} — {result.end_date}
            </span>
          )}
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
        <MetricCard
          label="Sharpe"
          value={result.sharpe.toFixed(2)}
          positive={result.sharpe > 0}
        />
        <MetricCard
          label="CAGR"
          value={(result.cagr * 100).toFixed(1)}
          suffix="%"
          positive={result.cagr > 0}
        />
        <MetricCard
          label="Max DD"
          value={(result.max_drawdown * 100).toFixed(1)}
          suffix="%"
          positive={result.max_drawdown > -0.15}
        />
        <MetricCard
          label="Win Rate"
          value={(result.win_rate * 100).toFixed(1)}
          suffix="%"
          positive={result.win_rate > 0.5}
        />
        <MetricCard label="Trades" value={result.n_trades} />
        <MetricCard
          label="Total Return"
          value={result.total_return_pct.toFixed(1)}
          suffix="%"
          positive={result.total_return_pct > 0}
        />
        <MetricCard
          label="Final Equity"
          value={`$${result.final_equity.toLocaleString()}`}
          positive={result.final_equity > initialCapital}
        />
      </div>

      <OHLCVChart
        result={result}
        initialCapital={initialCapital}
        trades={result.trades || []}
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="glass-effect">
          <CardHeader>
            <CardTitle className="text-sm">COSTS</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Total Fees</span>
              <span>${result.total_fees.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Total Funding</span>
              <span>${result.total_funding.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Avg Trade PnL</span>
              <span className={result.avg_trade_pnl > 0 ? 'text-green-400' : 'text-red-400'}>
                ${result.avg_trade_pnl.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Profit Factor</span>
              <span>{result.profit_factor.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Sortino</span>
              <span>{result.sortino.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Calmar</span>
              <span>{result.calmar.toFixed(2)}</span>
            </div>
          </CardContent>
        </Card>

        <RegimePanel regimeCounts={result.regime_counts} />

        <Card className="glass-effect">
          <CardHeader>
            <CardTitle className="text-sm">VS BASELINES</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            {result.baselines &&
              Object.entries(result.baselines).map(([name, metrics]) => (
                <div key={name} className="flex justify-between">
                  <span className="text-muted-foreground capitalize">
                    {name.replace(/_/g, ' ')}
                  </span>
                  <span
                    className={
                      metrics.total_return_pct > 0 ? 'text-green-400' : 'text-red-400'
                    }
                  >
                    {metrics.total_return_pct.toFixed(1)}%
                  </span>
                </div>
              ))}
            <div className="border-t border-border pt-2 mt-2 flex justify-between font-medium">
              <span>Strategy</span>
              <span
                className={
                  result.total_return_pct > 0 ? 'text-green-400' : 'text-red-400'
                }
              >
                {result.total_return_pct.toFixed(1)}%
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      <TradeTable trades={result.trades || []} />

      {result.score_reachability && (
        <Card className="glass-effect">
          <CardHeader>
            <CardTitle className="text-sm">SCORE DISTRIBUTION</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-5 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground block">Mean</span>
              {result.score_reachability.mean?.toFixed(2)}
            </div>
            <div>
              <span className="text-muted-foreground block">Std Dev</span>
              {result.score_reachability.std?.toFixed(2)}
            </div>
            <div>
              <span className="text-muted-foreground block">P5</span>
              {result.score_reachability.p5?.toFixed(2)}
            </div>
            <div>
              <span className="text-muted-foreground block">P95</span>
              {result.score_reachability.p95?.toFixed(2)}
            </div>
            <div>
              <span className="text-muted-foreground block">Entry %</span>
              <span
                className={
                  result.score_reachability.ok ? 'text-green-400' : 'text-red-400'
                }
              >
                {result.score_reachability.entry_pct?.toFixed(1)}%
                {result.score_reachability.ok ? ' OK' : ' WARNING'}
              </span>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
