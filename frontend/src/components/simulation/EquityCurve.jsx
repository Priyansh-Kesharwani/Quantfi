import React, { useMemo } from 'react';
import { formatCurrency } from '@/utils';
import { BarChart3 } from 'lucide-react';
import {
  ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
} from 'recharts';

const ASSET_COLORS = ['#EC4899', '#14B8A6', '#F97316', '#8B5CF6', '#06B6D4'];

const EquityCurve = ({ result }) => {
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

  if (!chartData.length) return null;

  return (
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
            {assetPriceSymbols.map((sym, i) => (
              <Line key={sym} type="monotone" dataKey={sym} name={sym}
                stroke={ASSET_COLORS[i % ASSET_COLORS.length]} strokeWidth={1.2}
                strokeDasharray="2 4" dot={false} connectNulls />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default EquityCurve;
