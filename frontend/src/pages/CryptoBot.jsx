import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Command, CommandInput, CommandList, CommandEmpty, CommandGroup, CommandItem } from '@/components/ui/command';
import { Calendar } from '@/components/ui/calendar';

import api from '@/api';
import { format, subDays, subMonths } from 'date-fns';
import { CalendarIcon, Search, ChevronDown, Wifi, WifiOff, ArrowUpRight, ArrowDownRight, Download, Filter, ArrowUpDown, Grid3X3 } from 'lucide-react';
import { toast } from 'sonner';

const TF_MS = { '5m': 300000, '15m': 900000, '1h': 3600000, '4h': 14400000, '1d': 86400000 };

const TF_DEFAULTS = {
  '5m':  { days: 14,   label: '2W' },
  '15m': { days: 30,   label: '1M' },
  '1h':  { days: 90,   label: '3M' },
  '4h':  { days: 180,  label: '6M' },
  '1d':  { days: 730,  label: '2Y' },
};

const DATE_PRESETS = [
  { label: '2W', days: 14 }, { label: '1M', days: 30 }, { label: '3M', days: 90 },
  { label: '6M', days: 180 }, { label: '1Y', days: 365 }, { label: '2Y', days: 730 },
  { label: '3Y', days: 1095 },
];

function computeBars(from, to, tf) {
  const ms = (to.getTime() - from.getTime());
  return Math.max(1, Math.floor(ms / (TF_MS[tf] || 3600000)));
}

function defaultDateRange(tf) {
  const d = TF_DEFAULTS[tf] || TF_DEFAULTS['1h'];
  const end = new Date();
  const start = subDays(end, d.days);
  return { from: start, to: end };
}

const MetricCard = ({ label, value, suffix = '', positive }) => (
  <div className="glass-effect p-3 rounded-lg">
    <p className="text-xs text-muted-foreground uppercase tracking-wider">{label}</p>
    <p className={`text-lg font-bold mt-1 ${positive === true ? 'text-green-400' : positive === false ? 'text-red-400' : 'text-foreground'}`}>
      {value}{suffix}
    </p>
  </div>
);

function CryptoAssetSearch({ value, onChange }) {
  const [open, setOpen] = useState(false);
  const [markets, setMarkets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    api.getCryptoMarkets()
      .then(({ data }) => {
        if (!mounted) return;
        setMarkets(data.markets || []);
        setOffline(!!data.offline);
      })
      .catch(() => mounted && setOffline(true))
      .finally(() => mounted && setLoading(false));
    return () => { mounted = false; };
  }, []);

  useEffect(() => {
    const handler = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setOpen(prev => !prev);
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, []);

  const grouped = useMemo(() => {
    const g = {};
    for (const m of markets) {
      const cat = m.category || 'Other';
      if (!g[cat]) g[cat] = [];
      g[cat].push(m);
    }
    return g;
  }, [markets]);

  const currentLabel = useMemo(() => {
    const m = markets.find(x => x.symbol === value);
    return m ? `${m.base}/USDT` : value.replace(':USDT', '');
  }, [markets, value]);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="outline" className="min-w-[180px] justify-between gap-2 font-mono text-sm glass-effect">
          <div className="flex items-center gap-2">
            <Search className="h-3.5 w-3.5 text-muted-foreground" />
            <span>{currentLabel}</span>
          </div>
          <div className="flex items-center gap-1.5">
            {offline ? <WifiOff className="h-3 w-3 text-yellow-500" /> : <Wifi className="h-3 w-3 text-green-500" />}
            <ChevronDown className="h-3 w-3 text-muted-foreground" />
          </div>
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[340px] p-0" align="start">
        <Command>
          <CommandInput placeholder="Search assets... (Cmd+K)" />
          <CommandList>
            {loading ? (
              <div className="py-6 text-center text-sm text-muted-foreground">Loading markets...</div>
            ) : (
              <>
                <CommandEmpty>No assets found.</CommandEmpty>
                {Object.entries(grouped).map(([cat, items]) => (
                  <CommandGroup key={cat} heading={cat}>
                    {items.map(m => (
                      <CommandItem key={m.symbol} value={`${m.base} ${m.name} ${m.symbol}`}
                        onSelect={() => { onChange(m.symbol); setOpen(false); }}
                        className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-medium text-sm">{m.base}</span>
                          <span className="text-xs text-muted-foreground">/USDT</span>
                        </div>
                        <div className="flex items-center gap-3 text-xs text-muted-foreground">
                          {m.price > 0 && <span>${m.price >= 1 ? m.price.toLocaleString(undefined, {maximumFractionDigits:2}) : m.price.toPrecision(4)}</span>}
                          {m.volume_24h > 0 && <span className="w-16 text-right">{formatVolume(m.volume_24h)}</span>}
                        </div>
                      </CommandItem>
                    ))}
                  </CommandGroup>
                ))}
              </>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

function formatVolume(v) {
  if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(0)}M`;
  if (v >= 1e3) return `$${(v / 1e3).toFixed(0)}K`;
  return `$${v.toFixed(0)}`;
}

function DateRangePicker({ dateRange, onDateRangeChange, timeframe }) {
  const [open, setOpen] = useState(false);
  const bars = computeBars(dateRange.from, dateRange.to, timeframe);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="text-xs text-muted-foreground">DATE RANGE</Label>
        <span className="text-xs font-mono text-muted-foreground px-1.5 py-0.5 rounded bg-muted">~{bars.toLocaleString()} bars</span>
      </div>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button variant="outline" className="w-full justify-start text-left font-normal glass-effect text-xs">
            <CalendarIcon className="mr-2 h-3.5 w-3.5" />
            {format(dateRange.from, 'MMM d, yyyy')} — {format(dateRange.to, 'MMM d, yyyy')}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0" align="start">
          <div className="flex gap-1 p-2 border-b flex-wrap">
            {DATE_PRESETS.map(p => (
              <Button key={p.label} variant="ghost" size="sm" className="h-7 text-xs px-2"
                onClick={() => { onDateRangeChange({ from: subDays(new Date(), p.days), to: new Date() }); }}>
                {p.label}
              </Button>
            ))}
          </div>
          <Calendar mode="range" selected={dateRange}
            onSelect={(range) => {
              if (range?.from && range?.to) onDateRangeChange(range);
              else if (range?.from) onDateRangeChange({ from: range.from, to: dateRange.to });
            }}
            numberOfMonths={2} disabled={{ after: new Date() }} defaultMonth={subMonths(new Date(), 1)} />
        </PopoverContent>
      </Popover>
    </div>
  );
}

function EquityChart({ result, initialCapital, trades = [] }) {
  const svgRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);

  const {
    pts, bhPts, pricePts, regimes, w, h, padL, padR, padT, padB,
    xScale, yMin, yMax, yScale, dateLabels, yLabels,
    pathD, bhPathD, pricePathD, drawdownPath, peakIdx, peakVal,
    pMin, pMax, priceYLabels, tradeMarkers,
  } = useMemo(() => {
    const pts = result.equity_curve || [];
    const bhPts = result.buy_hold_curve || [];
    const pricePts = result.price_curve || [];
    const regimes = result.regimes_sampled || [];
    if (pts.length < 2) return { pts, bhPts, pricePts, regimes, w: 800, h: 300, pathD: '' };

    const w = 800, h = 300;
    const padL = 60, padR = 55, padT = 25, padB = 35;
    const chartW = w - padL - padR;
    const chartH = h - padT - padB;

    const values = pts.map(p => p.equity);
    const bhValues = bhPts.map(p => p.equity);
    const allValues = [...values, ...bhValues, initialCapital];
    let yMin = Math.min(...allValues);
    let yMax = Math.max(...allValues);
    const yPad = (yMax - yMin) * 0.08 || 100;
    yMin -= yPad; yMax += yPad;
    const yRange = yMax - yMin || 1;

    const xScale = chartW / (pts.length - 1);
    const yScale = chartH / yRange;
    const toX = i => padL + i * xScale;
    const toY = v => padT + (yMax - v) * yScale;

    const priceValues = pricePts.map(p => p.price);
    let pMin = 0, pMax = 1;
    if (priceValues.length > 0) {
      pMin = Math.min(...priceValues); pMax = Math.max(...priceValues);
      const pPad = (pMax - pMin) * 0.08 || 1;
      pMin -= pPad; pMax += pPad;
    }
    const pRange = pMax - pMin || 1;
    const toYPrice = v => padT + (pMax - v) * (chartH / pRange);

    const pathD = pts.map((pt, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(pt.equity).toFixed(1)}`).join(' ');

    let bhPathD = '';
    if (bhPts.length > 1) {
      bhPathD = bhPts.map((pt, i) => {
        const xi = (i / (bhPts.length - 1)) * (pts.length - 1);
        return `${i === 0 ? 'M' : 'L'}${toX(xi).toFixed(1)},${toY(pt.equity).toFixed(1)}`;
      }).join(' ');
    }

    let pricePathD = '';
    if (pricePts.length > 1) {
      pricePathD = pricePts.map((pt, i) => {
        const xi = (i / (pricePts.length - 1)) * (pts.length - 1);
        return `${i === 0 ? 'M' : 'L'}${toX(xi).toFixed(1)},${toYPrice(pt.price).toFixed(1)}`;
      }).join(' ');
    }

    let runMax = values[0];
    const ddPoints = [];
    for (let i = 0; i < values.length; i++) {
      runMax = Math.max(runMax, values[i]);
      if (values[i] < runMax) ddPoints.push({ i, peak: runMax, val: values[i] });
    }
    let drawdownPath = '';
    if (ddPoints.length > 0) {
      let segments = [], seg = [ddPoints[0]];
      for (let j = 1; j < ddPoints.length; j++) {
        if (ddPoints[j].i === ddPoints[j - 1].i + 1) seg.push(ddPoints[j]);
        else { segments.push(seg); seg = [ddPoints[j]]; }
      }
      segments.push(seg);
      for (const s of segments) {
        let d = `M${toX(s[0].i).toFixed(1)},${toY(s[0].peak).toFixed(1)}`;
        for (const p of s) d += ` L${toX(p.i).toFixed(1)},${toY(p.peak).toFixed(1)}`;
        for (let k = s.length - 1; k >= 0; k--) d += ` L${toX(s[k].i).toFixed(1)},${toY(s[k].val).toFixed(1)}`;
        d += ' Z'; drawdownPath += d + ' ';
      }
    }

    let peakIdx = 0, peakVal = values[0];
    for (let i = 1; i < values.length; i++) { if (values[i] > peakVal) { peakVal = values[i]; peakIdx = i; } }

    const nLabels = Math.min(7, pts.length);
    const step = Math.floor(pts.length / nLabels);
    const dateLabels = [];
    for (let i = 0; i < pts.length; i += step) dateLabels.push({ x: toX(i), label: format(new Date(pts[i].date), 'MMM yyyy') });

    const niceStep = niceNum((yMax - yMin) / 5);
    const yLabels = [];
    let yl = Math.ceil(yMin / niceStep) * niceStep;
    while (yl <= yMax) { yLabels.push({ y: toY(yl), label: formatDollar(yl) }); yl += niceStep; }

    const priceYLabels = [];
    if (priceValues.length > 0) {
      const pStep = niceNum((pMax - pMin) / 5);
      let pl = Math.ceil(pMin / pStep) * pStep;
      while (pl <= pMax) { priceYLabels.push({ y: toYPrice(pl), label: formatPrice(pl) }); pl += pStep; }
    }

    const tradeMarkers = [];
    if (trades.length > 0 && pts.length > 1) {
      const minBar = trades.reduce((m, t) => Math.min(m, t.bar_idx || 0), Infinity);
      const maxBar = trades.reduce((m, t) => Math.max(m, t.bar_idx || 0), 0);
      const barRange = maxBar - minBar || 1;
      for (const t of trades) {
        if (t.side === 'long_entry' || t.side === 'short_entry' || t.side === 'long_exit' || t.side === 'short_exit') {
          const ptIdx = Math.round(((t.bar_idx - minBar) / barRange) * (pts.length - 1));
          const clampedIdx = Math.max(0, Math.min(pts.length - 1, ptIdx));
          const eq = pts[clampedIdx]?.equity;
          if (eq != null) {
            tradeMarkers.push({ x: toX(clampedIdx), y: toY(eq), isEntry: t.side.endsWith('_entry'), isLong: t.side.startsWith('long'), pnl: t.pnl || 0 });
          }
        }
      }
    }

    return { pts, bhPts, pricePts, regimes, w, h, padL, padR, padT, padB, xScale, yMin, yMax, yScale, dateLabels, yLabels, pathD, bhPathD, pricePathD, drawdownPath, peakIdx, peakVal, pMin, pMax, priceYLabels, tradeMarkers };
  }, [result, initialCapital, trades]);

  const handleMouseMove = useCallback((e) => {
    if (!svgRef.current || pts.length < 2) return;
    const rect = svgRef.current.getBoundingClientRect();
    const mouseX = ((e.clientX - rect.left) / rect.width) * w;
    const idx = Math.round(Math.max(0, Math.min(pts.length - 1, (mouseX - padL) / xScale)));
    const pt = pts[idx];
    if (!pt) return;
    const returnPct = ((pt.equity - initialCapital) / initialCapital * 100).toFixed(1);
    const d = new Date(pt.date);
    const priceIdx = pricePts.length > 0 ? Math.round(idx * (pricePts.length - 1) / (pts.length - 1)) : -1;
    const price = priceIdx >= 0 && priceIdx < pricePts.length ? pricePts[priceIdx].price : null;
    setTooltip({
      x: padL + idx * xScale,
      y: padT + (yMax - pt.equity) * ((h - padT - padB) / (yMax - yMin || 1)),
      date: format(d, 'MMM d, yyyy'), equity: formatDollar(pt.equity),
      price: price != null ? formatPrice(price) : null, returnPct, regime: regimes[idx] || '',
    });
  }, [pts, pricePts, regimes, w, h, padL, padT, padB, xScale, yMin, yMax, initialCapital]);

  if (!pts || pts.length < 2) return null;

  const toX = i => padL + i * xScale;
  const toY = v => padT + (yMax - v) * ((h - padT - padB) / (yMax - yMin || 1));
  const regimeColors = { TRENDING: 'rgba(59,130,246,0.08)', RANGING: 'rgba(234,179,8,0.06)', STRESS: 'rgba(239,68,68,0.08)' };

  const firstDate = format(new Date(pts[0].date), 'MMM yyyy');
  const lastDate = format(new Date(pts[pts.length - 1].date), 'MMM yyyy');
  const finalEquity = pts[pts.length - 1].equity;

  return (
    <Card className="glass-effect">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <CardTitle className="text-sm">EQUITY CURVE</CardTitle>
            <span className="text-xs text-muted-foreground">{firstDate} — {lastDate}</span>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1.5"><div className="w-3 h-0.5 bg-green-500 rounded" /><span className="text-muted-foreground">Strategy</span></div>
            {bhPathD && <div className="flex items-center gap-1.5"><div className="w-3 h-0.5 bg-gray-500 rounded" style={{ borderTop: '1px dashed #888' }} /><span className="text-muted-foreground">Buy & Hold</span></div>}
            {pricePathD && <div className="flex items-center gap-1.5"><div className="w-3 h-0.5 rounded" style={{ background: '#a78bfa' }} /><span className="text-muted-foreground">Price</span></div>}
            <div className="flex items-center gap-1.5"><div className="w-3 h-2 rounded-sm" style={{ background: 'rgba(239,68,68,0.2)' }} /><span className="text-muted-foreground">Drawdown</span></div>
            {tradeMarkers && tradeMarkers.length > 0 && <div className="flex items-center gap-1.5"><span className="text-green-400 text-[10px]">▲</span><span className="text-red-400 text-[10px]">▼</span><span className="text-muted-foreground">Trades</span></div>}
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <svg ref={svgRef} viewBox={`0 0 ${w} ${h}`} className="w-full" style={{ height: 320 }} preserveAspectRatio="xMidYMid meet"
          onMouseMove={handleMouseMove} onMouseLeave={() => setTooltip(null)}>
          <defs>
            <linearGradient id="eqGradUp" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#22c55e" stopOpacity="0.15" /><stop offset="100%" stopColor="#22c55e" stopOpacity="0.01" /></linearGradient>
            <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#ef4444" stopOpacity="0.02" /><stop offset="100%" stopColor="#ef4444" stopOpacity="0.12" /></linearGradient>
          </defs>
          {regimes.length > 0 && (() => { const bands = []; let curRegime = regimes[0], start = 0; for (let i = 1; i <= regimes.length; i++) { if (i === regimes.length || regimes[i] !== curRegime) { const x1 = toX(start); const x2 = toX(Math.min(i, pts.length - 1)); if (regimeColors[curRegime]) bands.push(<rect key={`r${start}`} x={x1} y={padT} width={x2 - x1} height={h - padT - padB} fill={regimeColors[curRegime]} />); if (i < regimes.length) { curRegime = regimes[i]; start = i; } } } return bands; })()}
          {yLabels && yLabels.map((yl, i) => <g key={`yl${i}`}><line x1={padL} y1={yl.y} x2={w - padR} y2={yl.y} stroke="#333" strokeWidth="0.5" /><text x={padL - 6} y={yl.y + 3.5} fill="#888" fontSize="9" textAnchor="end" fontFamily="monospace">{yl.label}</text></g>)}
          {dateLabels && dateLabels.map((dl, i) => <text key={`dl${i}`} x={dl.x} y={h - padB + 16} fill="#888" fontSize="9" textAnchor="middle" fontFamily="monospace">{dl.label}</text>)}
          <line x1={padL} y1={toY(initialCapital)} x2={w - padR} y2={toY(initialCapital)} stroke="#666" strokeWidth="0.5" strokeDasharray="4,4" />
          {drawdownPath && <path d={drawdownPath} fill="url(#ddGrad)" />}
          {bhPathD && <path d={bhPathD} fill="none" stroke="#888" strokeWidth="1" strokeDasharray="4,3" />}
          {pricePathD && <path d={pricePathD} fill="none" stroke="#a78bfa" strokeWidth="1" strokeOpacity="0.7" />}
          {priceYLabels && priceYLabels.map((pl, i) => <g key={`pl${i}`}><line x1={w - padR} y1={pl.y} x2={w - padR + 4} y2={pl.y} stroke="#a78bfa" strokeWidth="0.5" strokeOpacity="0.5" /><text x={w - padR + 7} y={pl.y + 3.5} fill="#a78bfa" fontSize="8" textAnchor="start" fontFamily="monospace" opacity="0.7">{pl.label}</text></g>)}
          {pathD && (() => { const lastX = toX(pts.length - 1); const baseY = h - padB; return <path d={pathD + ` L${lastX.toFixed(1)},${baseY} L${padL},${baseY} Z`} fill="url(#eqGradUp)" />; })()}
          <path d={pathD} fill="none" stroke={finalEquity >= initialCapital ? '#22c55e' : '#ef4444'} strokeWidth="1.5" />
          {peakIdx !== undefined && <circle cx={toX(peakIdx)} cy={toY(peakVal)} r="3" fill="#22c55e" stroke="#fff" strokeWidth="0.5" />}
          {tradeMarkers && tradeMarkers.map((m, i) => <g key={`tm${i}`}>{m.isEntry ? <polygon points={`${m.x},${m.y - 7} ${m.x - 4},${m.y} ${m.x + 4},${m.y}`} fill={m.isLong ? '#22c55e' : '#ef4444'} stroke="#000" strokeWidth="0.5" opacity="0.75" /> : <polygon points={`${m.x},${m.y + 7} ${m.x - 4},${m.y} ${m.x + 4},${m.y}`} fill={m.pnl >= 0 ? '#22c55e' : '#ef4444'} stroke="#000" strokeWidth="0.5" opacity="0.75" />}</g>)}
          {tooltip && <>
            <line x1={tooltip.x} y1={padT} x2={tooltip.x} y2={h - padB} stroke="#aaa" strokeWidth="0.5" strokeDasharray="2,2" />
            <line x1={padL} y1={tooltip.y} x2={w - padR} y2={tooltip.y} stroke="#aaa" strokeWidth="0.5" strokeDasharray="2,2" />
            <circle cx={tooltip.x} cy={tooltip.y} r="4" fill="#22c55e" stroke="#fff" strokeWidth="1" />
            <rect x={tooltip.x + 8} y={tooltip.y - 42} width="145" height={tooltip.price ? 60 : 48} rx="4" fill="rgba(0,0,0,0.88)" />
            <text x={tooltip.x + 14} y={tooltip.y - 28} fill="#ccc" fontSize="9" fontFamily="monospace">{tooltip.date}</text>
            <text x={tooltip.x + 14} y={tooltip.y - 16} fill="#22c55e" fontSize="10" fontWeight="bold" fontFamily="monospace">{tooltip.equity}</text>
            {tooltip.price && <text x={tooltip.x + 14} y={tooltip.y - 4} fill="#a78bfa" fontSize="9" fontFamily="monospace">Price: {tooltip.price}</text>}
            <text x={tooltip.x + 14} y={tooltip.price ? tooltip.y + 9 : tooltip.y - 3} fill={parseFloat(tooltip.returnPct) >= 0 ? '#22c55e' : '#ef4444'} fontSize="9" fontFamily="monospace">{parseFloat(tooltip.returnPct) >= 0 ? '+' : ''}{tooltip.returnPct}%{tooltip.regime && ` · ${tooltip.regime}`}</text>
          </>}
        </svg>
      </CardContent>
    </Card>
  );
}

function niceNum(range) {
  const exp = Math.floor(Math.log10(Math.abs(range) || 1));
  const frac = range / Math.pow(10, exp);
  let nice;
  if (frac <= 1.5) nice = 1; else if (frac <= 3) nice = 2; else if (frac <= 7) nice = 5; else nice = 10;
  return nice * Math.pow(10, exp);
}

function formatDollar(v) {
  if (Math.abs(v) >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  if (Math.abs(v) >= 1e3) return `$${(v / 1e3).toFixed(1)}K`;
  return `$${v.toFixed(0)}`;
}

function formatPrice(v) {
  if (v >= 10000) return `$${(v / 1000).toFixed(1)}K`;
  if (v >= 1) return `$${v.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
  return `$${v.toPrecision(4)}`;
}

const SIDE_LABELS = { long_entry: 'Long Entry', long_exit: 'Long Exit', short_entry: 'Short Entry', short_exit: 'Short Exit', grid_buy: 'Grid Buy', grid_sell: 'Grid Sell' };
const SIDE_COLORS = { long_entry: 'text-green-400', long_exit: 'text-green-600', short_entry: 'text-red-400', short_exit: 'text-red-600', grid_buy: 'text-blue-400', grid_sell: 'text-blue-600' };

function TradeTable({ trades = [] }) {
  const [sortKey, setSortKey] = useState('bar_idx');
  const [sortAsc, setSortAsc] = useState(true);
  const [sideFilter, setSideFilter] = useState('all');
  const [pnlFilter, setPnlFilter] = useState('all');
  const [page, setPage] = useState(0);
  const pageSize = 20;

  const filtered = useMemo(() => {
    let t = [...trades];
    if (sideFilter === 'long') t = t.filter(x => x.side.startsWith('long'));
    else if (sideFilter === 'short') t = t.filter(x => x.side.startsWith('short'));
    else if (sideFilter === 'grid') t = t.filter(x => x.side.startsWith('grid'));
    if (pnlFilter === 'winners') t = t.filter(x => x.pnl > 0);
    else if (pnlFilter === 'losers') t = t.filter(x => x.pnl < 0);
    return t;
  }, [trades, sideFilter, pnlFilter]);

  const sorted = useMemo(() => {
    const s = [...filtered];
    s.sort((a, b) => { let av = a[sortKey], bv = b[sortKey]; if (typeof av === 'string') av = av.toLowerCase(); if (typeof bv === 'string') bv = bv.toLowerCase(); if (av < bv) return sortAsc ? -1 : 1; if (av > bv) return sortAsc ? 1 : -1; return 0; });
    return s;
  }, [filtered, sortKey, sortAsc]);

  const pageCount = Math.ceil(sorted.length / pageSize);
  const pageRows = sorted.slice(page * pageSize, (page + 1) * pageSize);
  const toggleSort = (key) => { if (sortKey === key) setSortAsc(!sortAsc); else { setSortKey(key); setSortAsc(true); } };

  const totals = useMemo(() => {
    const pnls = filtered.filter(t => t.pnl !== 0).map(t => t.pnl);
    const wins = pnls.filter(p => p > 0);
    return { totalPnl: pnls.reduce((a, b) => a + b, 0), totalFees: filtered.reduce((a, t) => a + (t.fee || 0), 0), winRate: pnls.length > 0 ? (wins.length / pnls.length * 100) : 0, count: filtered.length };
  }, [filtered]);

  const exportCSV = useCallback(() => {
    const cols = ['timestamp', 'side', 'price', 'units', 'notional', 'pnl', 'fee', 'funding_paid', 'exit_reason', 'leverage', 'regime', 'bar_idx'];
    const header = cols.join(',');
    const rows = sorted.map(t => cols.map(c => { const v = t[c]; if (typeof v === 'string' && v.includes(',')) return `"${v}"`; if (typeof v === 'number') return Number(v.toFixed(4)); return v ?? ''; }).join(','));
    const csv = [header, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = `trades_${new Date().toISOString().slice(0, 10)}.csv`; a.click(); URL.revokeObjectURL(url);
  }, [sorted]);

  const SortHeader = ({ label, field, className = '' }) => (
    <th className={`px-2 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider cursor-pointer select-none hover:text-foreground transition-colors ${className}`} onClick={() => toggleSort(field)}>
      <span className="flex items-center gap-1">{label}{sortKey === field && <ArrowUpDown className="h-3 w-3" />}</span>
    </th>
  );

  if (trades.length === 0) return null;

  return (
    <Card className="glass-effect">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-3">
            <CardTitle className="text-sm">TRADE LOG</CardTitle>
            <span className="text-xs font-mono text-muted-foreground">{totals.count} trades · PnL: <span className={totals.totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}>${totals.totalPnl.toFixed(2)}</span>{' · '}Win: {totals.winRate.toFixed(0)}%</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <Filter className="h-3 w-3 text-muted-foreground" />
              <Select value={sideFilter} onValueChange={v => { setSideFilter(v); setPage(0); }}><SelectTrigger className="h-7 text-xs w-[100px] glass-effect"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="all">All Sides</SelectItem><SelectItem value="long">Long</SelectItem><SelectItem value="short">Short</SelectItem><SelectItem value="grid">Grid</SelectItem></SelectContent></Select>
              <Select value={pnlFilter} onValueChange={v => { setPnlFilter(v); setPage(0); }}><SelectTrigger className="h-7 text-xs w-[100px] glass-effect"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="all">All PnL</SelectItem><SelectItem value="winners">Winners</SelectItem><SelectItem value="losers">Losers</SelectItem></SelectContent></Select>
            </div>
            <Button variant="outline" size="sm" className="h-7 gap-1 text-xs" onClick={exportCSV}><Download className="h-3 w-3" /> CSV</Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead><tr className="border-b border-border"><SortHeader label="#" field="bar_idx" className="w-12" /><SortHeader label="Time" field="timestamp" /><SortHeader label="Side" field="side" /><SortHeader label="Price" field="price" /><SortHeader label="Size" field="notional" /><SortHeader label="PnL" field="pnl" /><SortHeader label="Fee" field="fee" /><SortHeader label="Regime" field="regime" /><SortHeader label="Exit Reason" field="exit_reason" /></tr></thead>
            <tbody>
              {pageRows.map((t, i) => (
                <tr key={i} className="border-b border-border/40 hover:bg-muted/30 transition-colors">
                  <td className="px-2 py-1.5 font-mono text-muted-foreground">{t.bar_idx}</td>
                  <td className="px-2 py-1.5 font-mono">{t.timestamp ? new Date(t.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—'}</td>
                  <td className="px-2 py-1.5"><span className={`flex items-center gap-1 font-medium ${SIDE_COLORS[t.side] || ''}`}>{t.side?.startsWith('long') || t.side === 'grid_buy' ? <ArrowUpRight className="h-3 w-3" /> : t.side?.startsWith('grid') ? <Grid3X3 className="h-3 w-3" /> : <ArrowDownRight className="h-3 w-3" />}{SIDE_LABELS[t.side] || t.side}</span></td>
                  <td className="px-2 py-1.5 font-mono">${t.price >= 1 ? t.price.toLocaleString(undefined, { maximumFractionDigits: 2 }) : t.price?.toPrecision(4)}</td>
                  <td className="px-2 py-1.5 font-mono">${t.notional?.toFixed(0)}</td>
                  <td className={`px-2 py-1.5 font-mono font-medium ${t.pnl > 0 ? 'text-green-400' : t.pnl < 0 ? 'text-red-400' : 'text-muted-foreground'}`}>{t.pnl === 0 ? '—' : `${t.pnl > 0 ? '+' : ''}$${t.pnl.toFixed(2)}`}</td>
                  <td className="px-2 py-1.5 font-mono text-muted-foreground">${t.fee?.toFixed(2)}</td>
                  <td className="px-2 py-1.5"><span className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${t.regime === 'TRENDING' ? 'bg-blue-500/20 text-blue-400' : t.regime === 'STRESS' ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'}`}>{t.regime}</span></td>
                  <td className="px-2 py-1.5 text-muted-foreground">{t.exit_reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {pageCount > 1 && (
          <div className="flex items-center justify-between mt-3 pt-2 border-t border-border">
            <span className="text-xs text-muted-foreground">Page {page + 1} of {pageCount} ({sorted.length} trades)</span>
            <div className="flex gap-1">
              <Button variant="outline" size="sm" className="h-6 px-2 text-xs" disabled={page === 0} onClick={() => setPage(p => p - 1)}>Prev</Button>
              <Button variant="outline" size="sm" className="h-6 px-2 text-xs" disabled={page >= pageCount - 1} onClick={() => setPage(p => p + 1)}>Next</Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function CryptoBot() {
  const [config, setConfig] = useState({ symbol: 'BTC/USDT:USDT', timeframe: '1h', initial_capital: 10000, strategy_mode: 'adaptive', leverage: 3, cost_preset: 'BINANCE_FUTURES_TAKER', entry_threshold: 30, exit_threshold: 15, max_holding_bars: 336, atr_trail_mult: 4.0, kelly_fraction: 0.25, max_risk_per_trade: 0.15, score_exit_patience: 3, grid_levels: 20, grid_spacing: 'geometric', grid_order_size: 100, atr_multiplier: 3.0 });
  const [dateRange, setDateRange] = useState(defaultDateRange('1h'));
  const [dateManual, setDateManual] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleTimeframeChange = useCallback((tf) => { setConfig(prev => ({ ...prev, timeframe: tf })); if (!dateManual) setDateRange(defaultDateRange(tf)); }, [dateManual]);
  const handleDateRangeChange = useCallback((range) => { setDateRange(range); setDateManual(true); }, []);

  const runBacktest = async () => {
    setLoading(true);
    try {
      const payload = { ...config, start_date: dateRange.from.toISOString().slice(0, 10), end_date: dateRange.to.toISOString().slice(0, 10) };
      const { data } = await api.runCryptoBacktest(payload);
      setResult(data);
      const src = data.data_source === 'live' ? ' (live data)' : ' (synthetic)';
      toast.success(`Backtest complete: ${data.n_trades} trades${src}`);
    } catch (err) { toast.error(err?.response?.data?.detail || 'Backtest failed'); } finally { setLoading(false); }
  };

  const updateConfig = (key, value) => setConfig(prev => ({ ...prev, [key]: value }));

  return (
    <div className="p-6 max-w-[1600px] mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div><h1 className="text-3xl font-bold tracking-tight">CRYPTO TRADING BOT</h1><p className="text-muted-foreground mt-1">Directional Futures + Grid Trading with Regime Detection</p></div>
          <CryptoAssetSearch value={config.symbol} onChange={(sym) => updateConfig('symbol', sym)} />
        </div>
        <Button onClick={runBacktest} disabled={loading} size="lg" className="min-w-[160px]">{loading ? 'RUNNING...' : 'RUN BACKTEST'}</Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <Card className="glass-effect lg:col-span-1">
          <CardHeader><CardTitle className="text-sm">CONFIGURATION</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <div><Label className="text-xs text-muted-foreground">STRATEGY</Label><Select value={config.strategy_mode} onValueChange={v => updateConfig('strategy_mode', v)}><SelectTrigger className="glass-effect mt-1"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="adaptive">Adaptive (Recommended)</SelectItem><SelectItem value="directional">Directional Futures</SelectItem><SelectItem value="grid">Grid Trading</SelectItem></SelectContent></Select></div>
            <div><Label className="text-xs text-muted-foreground">TIMEFRAME</Label><Select value={config.timeframe} onValueChange={handleTimeframeChange}><SelectTrigger className="glass-effect mt-1"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="5m">5 min</SelectItem><SelectItem value="15m">15 min</SelectItem><SelectItem value="1h">1 hour</SelectItem><SelectItem value="4h">4 hour</SelectItem><SelectItem value="1d">1 day</SelectItem></SelectContent></Select></div>
            <DateRangePicker dateRange={dateRange} onDateRangeChange={handleDateRangeChange} timeframe={config.timeframe} />
            <div><Label className="text-xs text-muted-foreground">LEVERAGE: {config.leverage}x</Label><Slider value={[config.leverage]} min={1} max={10} step={1} onValueChange={([v]) => updateConfig('leverage', v)} className="mt-2" /></div>
            <div><Label className="text-xs text-muted-foreground">INITIAL CAPITAL</Label><Input type="number" value={config.initial_capital} onChange={e => updateConfig('initial_capital', Number(e.target.value))} className="glass-effect mt-1" /></div>
            <div><Label className="text-xs text-muted-foreground">ENTRY THRESHOLD: {config.entry_threshold}</Label><Slider value={[config.entry_threshold]} min={10} max={60} step={5} onValueChange={([v]) => updateConfig('entry_threshold', v)} className="mt-2" /></div>
            <div><Label className="text-xs text-muted-foreground">EXIT THRESHOLD: {config.exit_threshold}</Label><Slider value={[config.exit_threshold]} min={5} max={30} step={5} onValueChange={([v]) => updateConfig('exit_threshold', v)} className="mt-2" /></div>
            <div><Label className="text-xs text-muted-foreground">ATR TRAIL: {config.atr_trail_mult}x</Label><Slider value={[config.atr_trail_mult]} min={2.0} max={6.0} step={0.5} onValueChange={([v]) => updateConfig('atr_trail_mult', v)} className="mt-2" /></div>
            <div><Label className="text-xs text-muted-foreground">RISK PER TRADE: {(config.max_risk_per_trade * 100).toFixed(0)}%</Label><Slider value={[config.max_risk_per_trade]} min={0.05} max={0.40} step={0.01} onValueChange={([v]) => updateConfig('max_risk_per_trade', v)} className="mt-2" /></div>
            <div><Label className="text-xs text-muted-foreground">KELLY FRACTION: {config.kelly_fraction}</Label><Slider value={[config.kelly_fraction]} min={0.1} max={0.5} step={0.05} onValueChange={([v]) => updateConfig('kelly_fraction', v)} className="mt-2" /></div>
            <div><Label className="text-xs text-muted-foreground">GRID LEVELS: {config.grid_levels}</Label><Slider value={[config.grid_levels]} min={5} max={40} step={5} onValueChange={([v]) => updateConfig('grid_levels', v)} className="mt-2" /></div>
            <div><Label className="text-xs text-muted-foreground">GRID ORDER SIZE: ${config.grid_order_size}</Label><Slider value={[config.grid_order_size]} min={50} max={500} step={50} onValueChange={([v]) => updateConfig('grid_order_size', v)} className="mt-2" /></div>
          </CardContent>
        </Card>

        <div className="lg:col-span-3 space-y-6">
          {result && <>
            {result.data_source && <div className="flex items-center gap-2 text-xs"><span className="px-2 py-0.5 rounded font-mono bg-green-500/20 text-green-400">LIVE DATA</span>{result.n_bars_actual > 0 && <span className="text-muted-foreground">{result.n_bars_actual.toLocaleString()} bars</span>}{result.start_date && result.end_date && <span className="text-muted-foreground">{result.start_date} — {result.end_date}</span>}</div>}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
              <MetricCard label="Sharpe" value={result.sharpe.toFixed(2)} positive={result.sharpe > 0} />
              <MetricCard label="CAGR" value={(result.cagr * 100).toFixed(1)} suffix="%" positive={result.cagr > 0} />
              <MetricCard label="Max DD" value={(result.max_drawdown * 100).toFixed(1)} suffix="%" positive={result.max_drawdown > -0.15} />
              <MetricCard label="Win Rate" value={(result.win_rate * 100).toFixed(1)} suffix="%" positive={result.win_rate > 0.5} />
              <MetricCard label="Trades" value={result.n_trades} />
              <MetricCard label="Total Return" value={result.total_return_pct.toFixed(1)} suffix="%" positive={result.total_return_pct > 0} />
              <MetricCard label="Final Equity" value={`$${result.final_equity.toLocaleString()}`} positive={result.final_equity > config.initial_capital} />
            </div>
            <EquityChart result={result} initialCapital={config.initial_capital} trades={result.trades || []} />
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card className="glass-effect"><CardHeader><CardTitle className="text-sm">COSTS</CardTitle></CardHeader><CardContent className="space-y-2 text-sm"><div className="flex justify-between"><span className="text-muted-foreground">Total Fees</span><span>${result.total_fees.toFixed(2)}</span></div><div className="flex justify-between"><span className="text-muted-foreground">Total Funding</span><span>${result.total_funding.toFixed(2)}</span></div><div className="flex justify-between"><span className="text-muted-foreground">Avg Trade PnL</span><span className={result.avg_trade_pnl > 0 ? 'text-green-400' : 'text-red-400'}>${result.avg_trade_pnl.toFixed(2)}</span></div><div className="flex justify-between"><span className="text-muted-foreground">Profit Factor</span><span>{result.profit_factor.toFixed(2)}</span></div><div className="flex justify-between"><span className="text-muted-foreground">Sortino</span><span>{result.sortino.toFixed(2)}</span></div><div className="flex justify-between"><span className="text-muted-foreground">Calmar</span><span>{result.calmar.toFixed(2)}</span></div></CardContent></Card>
              <Card className="glass-effect"><CardHeader><div className="flex items-center justify-between"><CardTitle className="text-sm">REGIMES</CardTitle><span className="text-xs px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 font-mono">HMM</span></div></CardHeader><CardContent className="space-y-3 text-sm">{result.regime_counts && (() => { const total = Object.values(result.regime_counts).reduce((a, b) => a + b, 0); const colors = { TRENDING: '#3b82f6', RANGING: '#eab308', STRESS: '#ef4444' }; return <>{<div className="flex h-3 rounded-full overflow-hidden gap-0.5">{Object.entries(result.regime_counts).map(([regime, count]) => <div key={regime} style={{ width: `${(count / total * 100).toFixed(1)}%`, backgroundColor: colors[regime] || '#666' }} className="rounded-sm" title={`${regime}: ${count}`} />)}</div>}{Object.entries(result.regime_counts).map(([regime, count]) => <div key={regime} className="flex justify-between"><span className={`font-medium ${regime === 'TRENDING' ? 'text-blue-400' : regime === 'RANGING' ? 'text-yellow-400' : 'text-red-400'}`}>{regime}</span><span>{count} bars ({(count / total * 100).toFixed(0)}%)</span></div>)}</>; })()}</CardContent></Card>
              <Card className="glass-effect"><CardHeader><CardTitle className="text-sm">VS BASELINES</CardTitle></CardHeader><CardContent className="space-y-2 text-sm">{result.baselines && Object.entries(result.baselines).map(([name, metrics]) => <div key={name} className="flex justify-between"><span className="text-muted-foreground capitalize">{name.replace(/_/g, ' ')}</span><span className={metrics.total_return_pct > 0 ? 'text-green-400' : 'text-red-400'}>{metrics.total_return_pct.toFixed(1)}%</span></div>)}<div className="border-t border-border pt-2 mt-2 flex justify-between font-medium"><span>Strategy</span><span className={result.total_return_pct > 0 ? 'text-green-400' : 'text-red-400'}>{result.total_return_pct.toFixed(1)}%</span></div></CardContent></Card>
            </div>
            <TradeTable trades={result.trades || []} />
            {result.score_reachability && <Card className="glass-effect"><CardHeader><CardTitle className="text-sm">SCORE DISTRIBUTION</CardTitle></CardHeader><CardContent className="grid grid-cols-5 gap-4 text-sm"><div><span className="text-muted-foreground block">Mean</span>{result.score_reachability.mean?.toFixed(2)}</div><div><span className="text-muted-foreground block">Std Dev</span>{result.score_reachability.std?.toFixed(2)}</div><div><span className="text-muted-foreground block">P5</span>{result.score_reachability.p5?.toFixed(2)}</div><div><span className="text-muted-foreground block">P95</span>{result.score_reachability.p95?.toFixed(2)}</div><div><span className="text-muted-foreground block">Entry %</span><span className={result.score_reachability.ok ? 'text-green-400' : 'text-red-400'}>{result.score_reachability.entry_pct?.toFixed(1)}%{result.score_reachability.ok ? ' OK' : ' WARNING'}</span></div></CardContent></Card>}
          </>}
          {!result && !loading && <Card className="glass-effect"><CardContent className="py-20 text-center"><p className="text-muted-foreground text-lg">Configure parameters and click RUN BACKTEST to start</p><p className="text-muted-foreground text-sm mt-2">The adaptive strategy switches between Directional Futures (trending markets) and Grid Trading (ranging markets)</p></CardContent></Card>}
          {loading && <Card className="glass-effect"><CardContent className="py-20 text-center"><div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-primary border-r-transparent"></div><p className="text-muted-foreground mt-4">Running backtest...</p></CardContent></Card>}
        </div>
      </div>
    </div>
  );
}
