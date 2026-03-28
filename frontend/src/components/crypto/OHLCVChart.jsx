import React, { useState, useCallback, useRef, useMemo } from 'react';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { format } from 'date-fns';

function niceNum(range) {
  const exp = Math.floor(Math.log10(Math.abs(range) || 1));
  const frac = range / Math.pow(10, exp);
  let nice;
  if (frac <= 1.5) nice = 1;
  else if (frac <= 3) nice = 2;
  else if (frac <= 7) nice = 5;
  else nice = 10;
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

const REGIME_BG = {
  TRENDING: 'rgba(59,130,246,0.08)',
  RANGING: 'rgba(234,179,8,0.06)',
  STRESS: 'rgba(239,68,68,0.08)',
};

export default function OHLCVChart({ result, initialCapital, trades = [] }) {
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

    const values = pts.map((p) => p.equity);
    const bhValues = bhPts.map((p) => p.equity);
    const allValues = [...values, ...bhValues, initialCapital];
    let yMin = Math.min(...allValues);
    let yMax = Math.max(...allValues);
    const yPad = (yMax - yMin) * 0.08 || 100;
    yMin -= yPad;
    yMax += yPad;
    const yRange = yMax - yMin || 1;

    const xScale = chartW / (pts.length - 1);
    const yScale = chartH / yRange;

    const toX = (i) => padL + i * xScale;
    const toY = (v) => padT + (yMax - v) * yScale;

    const priceValues = pricePts.map((p) => p.price);
    let pMin = 0,
      pMax = 1;
    if (priceValues.length > 0) {
      pMin = Math.min(...priceValues);
      pMax = Math.max(...priceValues);
      const pPad = (pMax - pMin) * 0.08 || 1;
      pMin -= pPad;
      pMax += pPad;
    }
    const pRange = pMax - pMin || 1;
    const toYPrice = (v) => padT + (pMax - v) * (chartH / pRange);

    const pathD = pts
      .map(
        (pt, i) =>
          `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(pt.equity).toFixed(1)}`
      )
      .join(' ');

    let bhPathD = '';
    if (bhPts.length > 1) {
      bhPathD = bhPts
        .map((pt, i) => {
          const xi = (i / (bhPts.length - 1)) * (pts.length - 1);
          return `${i === 0 ? 'M' : 'L'}${toX(xi).toFixed(1)},${toY(pt.equity).toFixed(1)}`;
        })
        .join(' ');
    }

    let pricePathD = '';
    if (pricePts.length > 1) {
      pricePathD = pricePts
        .map((pt, i) => {
          const xi = (i / (pricePts.length - 1)) * (pts.length - 1);
          return `${i === 0 ? 'M' : 'L'}${toX(xi).toFixed(1)},${toYPrice(pt.price).toFixed(1)}`;
        })
        .join(' ');
    }

    let runMax = values[0];
    const ddPoints = [];
    for (let i = 0; i < values.length; i++) {
      runMax = Math.max(runMax, values[i]);
      if (values[i] < runMax) {
        ddPoints.push({ i, peak: runMax, val: values[i] });
      }
    }
    let drawdownPath = '';
    if (ddPoints.length > 0) {
      let segments = [];
      let seg = [ddPoints[0]];
      for (let j = 1; j < ddPoints.length; j++) {
        if (ddPoints[j].i === ddPoints[j - 1].i + 1) {
          seg.push(ddPoints[j]);
        } else {
          segments.push(seg);
          seg = [ddPoints[j]];
        }
      }
      segments.push(seg);
      for (const s of segments) {
        let d = `M${toX(s[0].i).toFixed(1)},${toY(s[0].peak).toFixed(1)}`;
        for (const p of s) d += ` L${toX(p.i).toFixed(1)},${toY(p.peak).toFixed(1)}`;
        for (let k = s.length - 1; k >= 0; k--)
          d += ` L${toX(s[k].i).toFixed(1)},${toY(s[k].val).toFixed(1)}`;
        d += ' Z';
        drawdownPath += d + ' ';
      }
    }

    let peakIdx = 0,
      peakVal = values[0];
    for (let i = 1; i < values.length; i++) {
      if (values[i] > peakVal) {
        peakVal = values[i];
        peakIdx = i;
      }
    }

    const nLabels = Math.min(7, pts.length);
    const step = Math.floor(pts.length / nLabels);
    const dateLabels = [];
    for (let i = 0; i < pts.length; i += step) {
      const d = new Date(pts[i].date);
      dateLabels.push({ x: toX(i), label: format(d, 'MMM yyyy') });
    }

    const niceStep = niceNum((yMax - yMin) / 5);
    const yLabels = [];
    let yl = Math.ceil(yMin / niceStep) * niceStep;
    while (yl <= yMax) {
      yLabels.push({ y: toY(yl), label: formatDollar(yl) });
      yl += niceStep;
    }

    const priceYLabels = [];
    if (priceValues.length > 0) {
      const pStep = niceNum((pMax - pMin) / 5);
      let pl = Math.ceil(pMin / pStep) * pStep;
      while (pl <= pMax) {
        priceYLabels.push({ y: toYPrice(pl), label: formatPrice(pl) });
        pl += pStep;
      }
    }

    const tradeMarkers = [];
    if (trades.length > 0 && pts.length > 1) {
      const minBar = trades.reduce((m, t) => Math.min(m, t.bar_idx || 0), Infinity);
      const maxBar = trades.reduce((m, t) => Math.max(m, t.bar_idx || 0), 0);
      const barRange = maxBar - minBar || 1;
      for (const t of trades) {
        if (
          t.side === 'long_entry' ||
          t.side === 'short_entry' ||
          t.side === 'long_exit' ||
          t.side === 'short_exit'
        ) {
          const ptIdx = Math.round(
            ((t.bar_idx - minBar) / barRange) * (pts.length - 1)
          );
          const clampedIdx = Math.max(0, Math.min(pts.length - 1, ptIdx));
          const eq = pts[clampedIdx]?.equity;
          if (eq != null) {
            const isEntry = t.side.endsWith('_entry');
            const isLong = t.side.startsWith('long');
            tradeMarkers.push({
              x: toX(clampedIdx),
              y: toY(eq),
              isEntry,
              isLong,
              pnl: t.pnl || 0,
            });
          }
        }
      }
    }

    return {
      pts, bhPts, pricePts, regimes, w, h, padL, padR, padT, padB,
      xScale, yMin, yMax, yScale, dateLabels, yLabels,
      pathD, bhPathD, pricePathD, drawdownPath, peakIdx, peakVal,
      pMin, pMax, priceYLabels, tradeMarkers,
    };
  }, [result, initialCapital, trades]);

  const handleMouseMove = useCallback(
    (e) => {
      if (!svgRef.current || pts.length < 2) return;
      const rect = svgRef.current.getBoundingClientRect();
      const mouseX = ((e.clientX - rect.left) / rect.width) * w;
      const idx = Math.round(
        Math.max(0, Math.min(pts.length - 1, (mouseX - padL) / xScale))
      );
      const pt = pts[idx];
      if (!pt) return;
      const returnPct = (((pt.equity - initialCapital) / initialCapital) * 100).toFixed(1);
      const d = new Date(pt.date);
      const priceIdx =
        pricePts.length > 0
          ? Math.round((idx * (pricePts.length - 1)) / (pts.length - 1))
          : -1;
      const price =
        priceIdx >= 0 && priceIdx < pricePts.length ? pricePts[priceIdx].price : null;
      setTooltip({
        x: padL + idx * xScale,
        y:
          padT +
          (yMax - pt.equity) * ((h - padT - padB) / (yMax - yMin || 1)),
        date: format(d, 'MMM d, yyyy'),
        equity: formatDollar(pt.equity),
        price: price != null ? formatPrice(price) : null,
        returnPct,
        regime: regimes[idx] || '',
      });
    },
    [pts, pricePts, regimes, w, h, padL, padT, padB, xScale, yMin, yMax, initialCapital]
  );

  if (!pts || pts.length < 2) return null;

  const toX = (i) => padL + i * xScale;
  const toY = (v) => padT + (yMax - v) * ((h - padT - padB) / (yMax - yMin || 1));

  const firstDate = format(new Date(pts[0].date), 'MMM yyyy');
  const lastDate = format(new Date(pts[pts.length - 1].date), 'MMM yyyy');
  const finalEquity = pts[pts.length - 1].equity;
  const totalRetPct = (
    ((finalEquity - initialCapital) / initialCapital) *
    100
  ).toFixed(1);

  return (
    <Card className="glass-effect">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <CardTitle className="text-sm">EQUITY CURVE</CardTitle>
            <span className="text-xs text-muted-foreground">
              {firstDate} — {lastDate}
            </span>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-0.5 bg-green-500 rounded" />
              <span className="text-muted-foreground">Strategy</span>
            </div>
            {bhPathD && (
              <div className="flex items-center gap-1.5">
                <div
                  className="w-3 h-0.5 bg-gray-500 rounded"
                  style={{ borderTop: '1px dashed #888' }}
                />
                <span className="text-muted-foreground">Buy & Hold</span>
              </div>
            )}
            {pricePathD && (
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-0.5 rounded" style={{ background: '#a78bfa' }} />
                <span className="text-muted-foreground">Price</span>
              </div>
            )}
            <div className="flex items-center gap-1.5">
              <div
                className="w-3 h-2 rounded-sm"
                style={{ background: 'rgba(239,68,68,0.2)' }}
              />
              <span className="text-muted-foreground">Drawdown</span>
            </div>
            {tradeMarkers && tradeMarkers.length > 0 && (
              <div className="flex items-center gap-1.5">
                <span className="text-green-400 text-[10px]">▲</span>
                <span className="text-red-400 text-[10px]">▼</span>
                <span className="text-muted-foreground">Trades</span>
              </div>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${w} ${h}`}
          className="w-full"
          style={{ height: 320 }}
          preserveAspectRatio="xMidYMid meet"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setTooltip(null)}
        >
          <defs>
            <linearGradient id="eqGradUp" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#22c55e" stopOpacity="0.15" />
              <stop offset="100%" stopColor="#22c55e" stopOpacity="0.01" />
            </linearGradient>
            <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#ef4444" stopOpacity="0.02" />
              <stop offset="100%" stopColor="#ef4444" stopOpacity="0.12" />
            </linearGradient>
          </defs>

          {regimes.length > 0 &&
            (() => {
              const bands = [];
              let curRegime = regimes[0],
                start = 0;
              for (let i = 1; i <= regimes.length; i++) {
                if (i === regimes.length || regimes[i] !== curRegime) {
                  const x1 = toX(start);
                  const x2 = toX(Math.min(i, pts.length - 1));
                  if (REGIME_BG[curRegime]) {
                    bands.push(
                      <rect
                        key={`r${start}`}
                        x={x1}
                        y={padT}
                        width={x2 - x1}
                        height={h - padT - padB}
                        fill={REGIME_BG[curRegime]}
                      />
                    );
                  }
                  if (i < regimes.length) {
                    curRegime = regimes[i];
                    start = i;
                  }
                }
              }
              return bands;
            })()}

          {yLabels &&
            yLabels.map((yl, i) => (
              <g key={`yl${i}`}>
                <line
                  x1={padL}
                  y1={yl.y}
                  x2={w - padR}
                  y2={yl.y}
                  stroke="#333"
                  strokeWidth="0.5"
                />
                <text
                  x={padL - 6}
                  y={yl.y + 3.5}
                  fill="#888"
                  fontSize="9"
                  textAnchor="end"
                  fontFamily="monospace"
                >
                  {yl.label}
                </text>
              </g>
            ))}

          {dateLabels &&
            dateLabels.map((dl, i) => (
              <text
                key={`dl${i}`}
                x={dl.x}
                y={h - padB + 16}
                fill="#888"
                fontSize="9"
                textAnchor="middle"
                fontFamily="monospace"
              >
                {dl.label}
              </text>
            ))}

          <line
            x1={padL}
            y1={toY(initialCapital)}
            x2={w - padR}
            y2={toY(initialCapital)}
            stroke="#666"
            strokeWidth="0.5"
            strokeDasharray="4,4"
          />

          {drawdownPath && <path d={drawdownPath} fill="url(#ddGrad)" />}
          {bhPathD && (
            <path
              d={bhPathD}
              fill="none"
              stroke="#888"
              strokeWidth="1"
              strokeDasharray="4,3"
            />
          )}
          {pricePathD && (
            <path
              d={pricePathD}
              fill="none"
              stroke="#a78bfa"
              strokeWidth="1"
              strokeOpacity="0.7"
            />
          )}

          {priceYLabels &&
            priceYLabels.map((pl, i) => (
              <g key={`pl${i}`}>
                <line
                  x1={w - padR}
                  y1={pl.y}
                  x2={w - padR + 4}
                  y2={pl.y}
                  stroke="#a78bfa"
                  strokeWidth="0.5"
                  strokeOpacity="0.5"
                />
                <text
                  x={w - padR + 7}
                  y={pl.y + 3.5}
                  fill="#a78bfa"
                  fontSize="8"
                  textAnchor="start"
                  fontFamily="monospace"
                  opacity="0.7"
                >
                  {pl.label}
                </text>
              </g>
            ))}

          {pathD &&
            (() => {
              const lastX = toX(pts.length - 1);
              const baseY = h - padB;
              const areaD = pathD + ` L${lastX.toFixed(1)},${baseY} L${padL},${baseY} Z`;
              return <path d={areaD} fill="url(#eqGradUp)" />;
            })()}

          <path
            d={pathD}
            fill="none"
            stroke={finalEquity >= initialCapital ? '#22c55e' : '#ef4444'}
            strokeWidth="1.5"
          />

          {peakIdx !== undefined && (
            <circle
              cx={toX(peakIdx)}
              cy={toY(peakVal)}
              r="3"
              fill="#22c55e"
              stroke="#fff"
              strokeWidth="0.5"
            />
          )}

          {tradeMarkers &&
            tradeMarkers.map((m, i) => (
              <g key={`tm${i}`}>
                {m.isEntry ? (
                  <polygon
                    points={`${m.x},${m.y - 7} ${m.x - 4},${m.y} ${m.x + 4},${m.y}`}
                    fill={m.isLong ? '#22c55e' : '#ef4444'}
                    stroke="#000"
                    strokeWidth="0.5"
                    opacity="0.75"
                  />
                ) : (
                  <polygon
                    points={`${m.x},${m.y + 7} ${m.x - 4},${m.y} ${m.x + 4},${m.y}`}
                    fill={m.pnl >= 0 ? '#22c55e' : '#ef4444'}
                    stroke="#000"
                    strokeWidth="0.5"
                    opacity="0.75"
                  />
                )}
              </g>
            ))}

          {tooltip && (
            <>
              <line
                x1={tooltip.x}
                y1={padT}
                x2={tooltip.x}
                y2={h - padB}
                stroke="#aaa"
                strokeWidth="0.5"
                strokeDasharray="2,2"
              />
              <line
                x1={padL}
                y1={tooltip.y}
                x2={w - padR}
                y2={tooltip.y}
                stroke="#aaa"
                strokeWidth="0.5"
                strokeDasharray="2,2"
              />
              <circle
                cx={tooltip.x}
                cy={tooltip.y}
                r="4"
                fill="#22c55e"
                stroke="#fff"
                strokeWidth="1"
              />
              <rect
                x={tooltip.x + 8}
                y={tooltip.y - 42}
                width="145"
                height={tooltip.price ? 60 : 48}
                rx="4"
                fill="rgba(0,0,0,0.88)"
              />
              <text
                x={tooltip.x + 14}
                y={tooltip.y - 28}
                fill="#ccc"
                fontSize="9"
                fontFamily="monospace"
              >
                {tooltip.date}
              </text>
              <text
                x={tooltip.x + 14}
                y={tooltip.y - 16}
                fill="#22c55e"
                fontSize="10"
                fontWeight="bold"
                fontFamily="monospace"
              >
                {tooltip.equity}
              </text>
              {tooltip.price && (
                <text
                  x={tooltip.x + 14}
                  y={tooltip.y - 4}
                  fill="#a78bfa"
                  fontSize="9"
                  fontFamily="monospace"
                >
                  Price: {tooltip.price}
                </text>
              )}
              <text
                x={tooltip.x + 14}
                y={tooltip.price ? tooltip.y + 9 : tooltip.y - 3}
                fill={parseFloat(tooltip.returnPct) >= 0 ? '#22c55e' : '#ef4444'}
                fontSize="9"
                fontFamily="monospace"
              >
                {parseFloat(tooltip.returnPct) >= 0 ? '+' : ''}
                {tooltip.returnPct}%{tooltip.regime && ` · ${tooltip.regime}`}
              </text>
            </>
          )}
        </svg>
      </CardContent>
    </Card>
  );
}
