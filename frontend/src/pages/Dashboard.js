import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api';
import { formatCurrency, getScoreColor, getZoneLabel } from '../utils';
import {
  RefreshCw, TrendingUp, TrendingDown, AlertCircle, Wallet,
  Activity, Target, Zap, ChevronRight, ArrowUpRight,
  ArrowDownRight, BarChart3, Shield, Eye, Newspaper,
  PieChart as PieChartIcon, Flame, Gauge
} from 'lucide-react';
import { toast } from 'sonner';
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis,
  Tooltip, BarChart, Bar, Cell, RadialBarChart, RadialBar,
  PieChart, Pie
} from 'recharts';

/* ─── helpers ─── */
const fmt = (v) => {
  if (v >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `$${(v / 1e3).toFixed(1)}K`;
  return `$${v.toFixed(2)}`;
};

const fmtINR = (v) => {
  if (v >= 1e7) return `₹${(v / 1e7).toFixed(2)}Cr`;
  if (v >= 1e5) return `₹${(v / 1e5).toFixed(2)}L`;
  if (v >= 1e3) return `₹${(v / 1e3).toFixed(1)}K`;
  return `₹${v.toFixed(2)}`;
};

const ZONE_COLORS = {
  strong_buy: '#22C55E',
  favorable: '#10B981',
  neutral: '#F59E0B',
  unfavorable: '#EF4444',
};

const TYPE_COLORS = {
  equity: '#6366F1',
  etf: '#8B5CF6',
  commodity: '#F59E0B',
  crypto: '#EC4899',
  index: '#06B6D4',
  other: '#94A3B8',
};

/* ─── KPI Stat Card ─── */
const StatCard = ({ icon: Icon, label, value, subValue, trend, trendPositive, className = '' }) => (
  <div className={`glass-effect rounded-sm p-5 ${className}`}>
    <div className="flex items-center gap-2 mb-3">
      <div className="p-2 rounded bg-white/5">
        <Icon className="w-4 h-4 text-primary" />
      </div>
      <span className="text-xs text-muted-foreground uppercase tracking-wider">{label}</span>
    </div>
    <div className="text-2xl font-bold font-data">{value}</div>
    <div className="flex items-center gap-2 mt-1">
      {subValue && <span className="text-xs text-muted-foreground font-data">{subValue}</span>}
      {trend != null && (
        <span className={`flex items-center gap-0.5 text-xs font-bold ${trendPositive ? 'text-emerald-400' : 'text-red-400'}`}>
          {trendPositive ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
          {trend}
        </span>
      )}
    </div>
  </div>
);

/* ─── DCA Opportunity Radar (Novel Feature) ─── */
const OpportunityRadar = ({ assets }) => {
  const radarData = useMemo(() => {
    return assets
      .filter(item => item.score?.composite_score != null)
      .map(item => ({
        name: item.asset.symbol,
        score: item.score.composite_score,
        fill: ZONE_COLORS[item.score?.zone] || ZONE_COLORS.neutral,
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 8);
  }, [assets]);

  if (radarData.length === 0) return null;

  return (
    <div className="glass-effect rounded-sm p-5">
      <div className="flex items-center gap-2 mb-4">
        <Target className="w-5 h-5 text-primary" />
        <h3 className="text-lg font-bold">DCA OPPORTUNITY RADAR</h3>
      </div>
      <div className="flex items-center gap-6">
        <div style={{ width: 180, height: 180 }}>
          <ResponsiveContainer>
            <RadialBarChart
              cx="50%"
              cy="50%"
              innerRadius="20%"
              outerRadius="90%"
              data={radarData}
              startAngle={180}
              endAngle={0}
              barSize={12}
            >
              <RadialBar
                background={{ fill: 'rgba(255,255,255,0.03)' }}
                dataKey="score"
                cornerRadius={4}
              />
            </RadialBarChart>
          </ResponsiveContainer>
        </div>
        <div className="flex-1 space-y-2">
          {radarData.map((item) => (
            <div key={item.name} className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: item.fill }} />
                <span className="text-xs font-bold font-data">{item.name}</span>
              </div>
              <span className="text-xs font-data font-bold" style={{ color: item.fill }}>
                {item.score.toFixed(0)}
              </span>
            </div>
          ))}
        </div>
      </div>
      <div className="mt-3 pt-3 border-t border-white/5 text-[10px] text-muted-foreground">
        Higher scores = better DCA accumulation opportunity
      </div>
    </div>
  );
};

/* ─── Portfolio Allocation Donut ─── */
const AllocationDonut = ({ assets }) => {
  const data = useMemo(() => {
    const typeMap = {};
    assets.forEach(item => {
      const t = item.asset?.asset_type || 'other';
      const val = item.price?.price_usd || 0;
      typeMap[t] = (typeMap[t] || 0) + val;
    });
    return Object.entries(typeMap).map(([name, value]) => ({
      name: name.charAt(0).toUpperCase() + name.slice(1),
      value,
      fill: TYPE_COLORS[name] || TYPE_COLORS.other,
    }));
  }, [assets]);

  if (data.length === 0) return null;

  return (
    <div className="glass-effect rounded-sm p-5">
      <div className="flex items-center gap-2 mb-4">
        <PieChartIcon className="w-5 h-5 text-primary" />
        <h3 className="text-lg font-bold">PORTFOLIO ALLOCATION</h3>
      </div>
      <div className="flex items-center gap-6">
        <div style={{ width: 140, height: 140 }}>
          <ResponsiveContainer>
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                innerRadius={35}
                outerRadius={60}
                paddingAngle={3}
                dataKey="value"
                strokeWidth={0}
              >
                {data.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="flex-1 space-y-2">
          {data.map(item => (
            <div key={item.name} className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: item.fill }} />
                <span className="text-xs">{item.name}</span>
              </div>
              <span className="text-xs font-data text-muted-foreground">{fmt(item.value)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

/* ─── Portfolio Health Score (Novel) ─── */
const PortfolioHealth = ({ avgScore, strongBuyCount, favorableCount, totalCount }) => {
  const health = Math.min(100, Math.round(avgScore * 1.2 + (strongBuyCount / Math.max(totalCount, 1)) * 20));
  const healthColor = health >= 70 ? '#22C55E' : health >= 45 ? '#F59E0B' : '#EF4444';
  const healthLabel = health >= 70 ? 'EXCELLENT' : health >= 45 ? 'MODERATE' : 'WEAK';

  return (
    <div className="glass-effect rounded-sm p-5">
      <div className="flex items-center gap-2 mb-4">
        <Shield className="w-5 h-5 text-primary" />
        <h3 className="text-lg font-bold">PORTFOLIO HEALTH</h3>
      </div>
      <div className="flex items-center gap-6">
        <div className="relative">
          <svg width="100" height="100" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="8" />
            <circle
              cx="50" cy="50" r="42"
              fill="none" stroke={healthColor} strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={`${(health / 100) * 264} 264`}
              transform="rotate(-90 50 50)"
              style={{ transition: 'stroke-dasharray 0.8s ease' }}
            />
            <text x="50" y="46" textAnchor="middle" fill={healthColor} fontSize="22" fontWeight="bold" fontFamily="JetBrains Mono">
              {health}
            </text>
            <text x="50" y="62" textAnchor="middle" fill="rgba(255,255,255,0.5)" fontSize="8" fontWeight="600">
              {healthLabel}
            </text>
          </svg>
        </div>
        <div className="flex-1 space-y-3">
          <div>
            <div className="text-[10px] text-muted-foreground">STRONG BUY ZONES</div>
            <div className="text-sm font-bold font-data text-emerald-400">{strongBuyCount} assets</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground">FAVORABLE ZONES</div>
            <div className="text-sm font-bold font-data text-teal-400">{favorableCount} assets</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground">AVG DCA SCORE</div>
            <div className="text-sm font-bold font-data" style={{ color: getScoreColor(avgScore) }}>{avgScore.toFixed(1)}</div>
          </div>
        </div>
      </div>
    </div>
  );
};

/* ─── Recent News Widget ─── */
const RecentNewsMini = () => {
  const [news, setNews] = useState([]);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await api.getNews(5);
        setNews((res.data.news || []).slice(0, 4));
      } catch (e) {
        /* silent */
      }
    };
    load();
  }, []);

  const formatAge = (dt) => {
    if (!dt) return '';
    const h = Math.floor((Date.now() - new Date(dt).getTime()) / 3600000);
    if (h < 1) return 'Just now';
    if (h < 24) return `${h}h ago`;
    return `${Math.floor(h / 24)}d ago`;
  };

  return (
    <div className="glass-effect rounded-sm p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Newspaper className="w-5 h-5 text-primary" />
          <h3 className="text-lg font-bold">RECENT NEWS</h3>
        </div>
        <a href="/news" className="text-xs text-primary hover:underline flex items-center gap-1">
          VIEW ALL <ChevronRight className="w-3 h-3" />
        </a>
      </div>
      {news.length === 0 ? (
        <p className="text-xs text-muted-foreground">No news available. Click refresh to fetch headlines.</p>
      ) : (
        <div className="space-y-3">
          {news.map((article, i) => (
            <div key={i} className="flex items-start gap-3 group">
              <div className={`mt-1 w-2 h-2 rounded-full shrink-0 ${
                article.event_type === 'rate_change' || article.event_type === 'war'
                  ? 'bg-red-400' : article.event_type === 'earnings'
                  ? 'bg-emerald-400' : 'bg-blue-400'
              }`} />
              <div className="flex-1 min-w-0">
                <p className="text-xs leading-tight line-clamp-2 group-hover:text-foreground transition text-foreground/80">
                  {article.title}
                </p>
                <div className="flex items-center gap-2 mt-1">
                  {article.source && <span className="text-[10px] text-muted-foreground">{article.source}</span>}
                  <span className="text-[10px] text-muted-foreground">{formatAge(article.published_at)}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

/* ─── Top/Bottom Performers Bar Chart ─── */
const PerformanceChart = ({ assets }) => {
  const chartData = useMemo(() => {
    return assets
      .filter(item => item.score?.composite_score != null)
      .map(item => ({
        symbol: item.asset.symbol,
        score: item.score.composite_score,
        fill: ZONE_COLORS[item.score?.zone] || ZONE_COLORS.neutral,
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 8);
  }, [assets]);

  if (chartData.length === 0) return null;

  return (
    <div className="glass-effect rounded-sm p-5">
      <div className="flex items-center gap-2 mb-4">
        <BarChart3 className="w-5 h-5 text-primary" />
        <h3 className="text-lg font-bold">TOP ASSET PERFORMANCE</h3>
      </div>
      <div style={{ width: '100%', height: 220 }}>
        <ResponsiveContainer>
          <BarChart data={chartData} margin={{ top: 5, right: 5, left: -10, bottom: 5 }}>
            <XAxis
              dataKey="symbol"
              tick={{ fontSize: 10, fill: '#888' }}
              axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
              tickLine={false}
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fontSize: 10, fill: '#888' }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0].payload;
                return (
                  <div className="glass-effect rounded p-2 text-xs border border-white/10">
                    <p className="font-bold">{d.symbol}</p>
                    <p style={{ color: d.fill }}>Score: {d.score.toFixed(1)}</p>
                  </div>
                );
              }}
            />
            <Bar dataKey="score" radius={[4, 4, 0, 0]} barSize={32}>
              {chartData.map((entry, i) => (
                <Cell key={i} fill={entry.fill} fillOpacity={0.8} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

/* ─── Smart Alerts (Novel) ─── */
const SmartAlerts = ({ assets }) => {
  const alerts = useMemo(() => {
    const result = [];
    assets.forEach(item => {
      const score = item.score?.composite_score;
      const zone = item.score?.zone;
      const rsi = item.indicators?.rsi_14;
      const drawdown = item.indicators?.drawdown_pct;

      if (zone === 'strong_buy') {
        result.push({
          type: 'opportunity',
          symbol: item.asset.symbol,
          message: `Strong buy zone — DCA score ${score?.toFixed(0)}`,
          icon: Flame,
          color: 'text-emerald-400 bg-emerald-400/10',
        });
      }
      if (rsi != null && rsi < 30) {
        result.push({
          type: 'oversold',
          symbol: item.asset.symbol,
          message: `RSI ${rsi.toFixed(1)} — Oversold territory`,
          icon: Activity,
          color: 'text-blue-400 bg-blue-400/10',
        });
      }
      if (drawdown != null && drawdown < -20) {
        result.push({
          type: 'drawdown',
          symbol: item.asset.symbol,
          message: `${drawdown.toFixed(1)}% drawdown from highs`,
          icon: TrendingDown,
          color: 'text-red-400 bg-red-400/10',
        });
      }
    });
    return result.slice(0, 5);
  }, [assets]);

  if (alerts.length === 0) return null;

  return (
    <div className="glass-effect rounded-sm p-5">
      <div className="flex items-center gap-2 mb-4">
        <Zap className="w-5 h-5 text-primary" />
        <h3 className="text-lg font-bold">SMART ALERTS</h3>
        <span className="ml-auto text-[10px] px-2 py-0.5 rounded-full bg-primary/20 text-primary font-bold">
          {alerts.length}
        </span>
      </div>
      <div className="space-y-2">
        {alerts.map((alert, i) => {
          const Icon = alert.icon;
          return (
            <div key={i} className={`flex items-center gap-3 p-3 rounded ${alert.color}`}>
              <Icon className="w-4 h-4 shrink-0" />
              <div className="flex-1 min-w-0">
                <span className="text-xs font-bold font-data">{alert.symbol}</span>
                <span className="text-xs ml-2 opacity-80">{alert.message}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

/* ─── Quick Asset Glance Row ─── */
const QuickGlance = ({ assets, navigate }) => {
  const topAssets = useMemo(() => {
    return assets
      .filter(item => item.score?.composite_score != null)
      .sort((a, b) => (b.score?.composite_score || 0) - (a.score?.composite_score || 0))
      .slice(0, 5);
  }, [assets]);

  return (
    <div className="glass-effect rounded-sm p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Eye className="w-5 h-5 text-primary" />
          <h3 className="text-lg font-bold">WATCHLIST QUICK GLANCE</h3>
        </div>
        <a href="/assets" className="text-xs text-primary hover:underline flex items-center gap-1">
          VIEW ALL <ChevronRight className="w-3 h-3" />
        </a>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-white/5">
              <th className="text-left text-[10px] text-muted-foreground pb-2 uppercase tracking-wider">Asset</th>
              <th className="text-right text-[10px] text-muted-foreground pb-2 uppercase tracking-wider">Price</th>
              <th className="text-right text-[10px] text-muted-foreground pb-2 uppercase tracking-wider">DCA Score</th>
              <th className="text-right text-[10px] text-muted-foreground pb-2 uppercase tracking-wider">Zone</th>
              <th className="text-right text-[10px] text-muted-foreground pb-2 uppercase tracking-wider">RSI</th>
              <th className="text-right text-[10px] text-muted-foreground pb-2 uppercase tracking-wider">Drawdown</th>
            </tr>
          </thead>
          <tbody>
            {topAssets.map(item => {
              const zone = item.score?.zone || 'neutral';
              return (
                <tr
                  key={item.asset.symbol}
                  className="border-b border-white/[0.03] hover:bg-white/[0.03] cursor-pointer transition"
                  onClick={() => navigate(`/assets/${item.asset.symbol}`)}
                >
                  <td className="py-3">
                    <div>
                      <span className="text-sm font-bold font-data">{item.asset.symbol}</span>
                      <span className="text-xs text-muted-foreground ml-2 hidden sm:inline">{item.asset.name}</span>
                    </div>
                  </td>
                  <td className="text-right py-3">
                    <span className="text-sm font-data">{formatCurrency(item.price?.price_usd, 'USD')}</span>
                  </td>
                  <td className="text-right py-3">
                    <span className="text-sm font-data font-bold" style={{ color: getScoreColor(item.score?.composite_score || 0) }}>
                      {(item.score?.composite_score || 0).toFixed(0)}
                    </span>
                  </td>
                  <td className="text-right py-3">
                    <span className={`text-[10px] px-2 py-0.5 rounded font-bold score-zone-${zone}`}>
                      {getZoneLabel(zone)}
                    </span>
                  </td>
                  <td className="text-right py-3">
                    <span className="text-xs font-data">
                      {item.indicators?.rsi_14 != null ? item.indicators.rsi_14.toFixed(1) : '—'}
                    </span>
                  </td>
                  <td className="text-right py-3">
                    <span className={`text-xs font-data ${(item.indicators?.drawdown_pct || 0) < -10 ? 'text-red-400' : ''}`}>
                      {item.indicators?.drawdown_pct != null ? `${item.indicators.drawdown_pct.toFixed(1)}%` : '—'}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

/* ──────────────────────────────
   MAIN DASHBOARD COMPONENT
   ────────────────────────────── */
const Dashboard = ({ refreshKey = 0 }) => {
  const navigate = useNavigate();
  const [dashboardData, setDashboardData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchDashboard = async () => {
    try {
      const response = await api.getDashboard();
      const assets = response.data.assets || [];
      setDashboardData(assets);
    } catch (error) {
      console.error('[Dashboard] Error fetching:', error);
      toast.error('Failed to load dashboard data');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchDashboard();
  }, [refreshKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleRefresh = () => {
    setRefreshing(true);
    fetchDashboard();
  };
  
  /* ─── Computed Portfolio Metrics ─── */
  const metrics = useMemo(() => {
    let totalUSD = 0;
    let totalINR = 0;
    let scoreSum = 0;
    let scoreCount = 0;
    let bestAsset = null;
    let worstAsset = null;
    let strongBuyCount = 0;
    let favorableCount = 0;

    dashboardData.forEach(item => {
      const priceUSD = item.price?.price_usd || 0;
      const priceINR = item.price?.price_inr || 0;
      totalUSD += priceUSD;
      totalINR += priceINR;

      const score = item.score?.composite_score;
      if (score != null) {
        scoreSum += score;
        scoreCount++;
        if (!bestAsset || score > bestAsset.score) bestAsset = { symbol: item.asset.symbol, score };
        if (!worstAsset || score < worstAsset.score) worstAsset = { symbol: item.asset.symbol, score };
      }

      if (item.score?.zone === 'strong_buy') strongBuyCount++;
      if (item.score?.zone === 'favorable') favorableCount++;
    });

    const avgScore = scoreCount > 0 ? scoreSum / scoreCount : 0;

    return {
      totalUSD, totalINR, avgScore, scoreCount,
      bestAsset, worstAsset,
      strongBuyCount, favorableCount,
      totalAssets: dashboardData.length,
    };
  }, [dashboardData]);

  /* ─── Loading ─── */
  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen" data-testid="loading-dashboard">
        <div className="text-center">
          <RefreshCw className="w-12 h-12 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  /* ─── Empty State ─── */
  if (dashboardData.length === 0) {
    return (
      <div className="flex items-center justify-center h-screen" data-testid="empty-dashboard">
        <div className="text-center max-w-md">
          <TrendingUp className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
          <h2 className="text-2xl font-bold mb-2">NO ASSETS IN WATCHLIST</h2>
          <p className="text-muted-foreground mb-6">
            Add assets to your watchlist to start analyzing DCA opportunities.
          </p>
          <button
            className="px-6 py-3 bg-primary text-primary-foreground rounded font-medium hover:bg-primary/90 transition"
            onClick={() => window.dispatchEvent(new Event('addAsset'))}
            data-testid="add-first-asset-btn"
          >
            ADD YOUR FIRST ASSET
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 lg:p-8 max-w-[1600px] mx-auto" data-testid="dashboard-main">
      {/* ─── Header ─── */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-4xl font-bold tracking-tight mb-1" data-testid="dashboard-title">DASHBOARD</h1>
          <p className="text-muted-foreground text-sm">Real-time DCA intelligence overview</p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="flex items-center gap-2 px-4 py-2 glass-effect hover:bg-white/10 rounded transition"
          data-testid="refresh-dashboard-btn"
        >
          <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
          <span className="text-sm">REFRESH</span>
        </button>
      </div>

      {/* ─── KPI Stats Row ─── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <StatCard
          icon={Wallet}
          label="Portfolio Snapshot"
          value={fmt(metrics.totalUSD)}
          subValue={fmtINR(metrics.totalINR)}
        />
        <StatCard
          icon={Gauge}
          label="Avg DCA Score"
          value={metrics.avgScore.toFixed(1)}
          subValue={`${metrics.scoreCount} assets scored`}
          trend={metrics.avgScore >= 50 ? 'Favorable' : 'Below avg'}
          trendPositive={metrics.avgScore >= 50}
        />
        <StatCard
          icon={TrendingUp}
          label="Best Opportunity"
          value={metrics.bestAsset?.symbol || '—'}
          subValue={metrics.bestAsset ? `Score: ${metrics.bestAsset.score.toFixed(0)}` : 'N/A'}
          trendPositive={true}
          trend={metrics.bestAsset?.score >= 60 ? 'Buy zone' : null}
        />
        <StatCard
          icon={Activity}
          label="Total Assets"
          value={metrics.totalAssets}
          subValue={`${metrics.strongBuyCount} in buy zone`}
        />
      </div>

      {/* ─── Main Grid: Charts + Insights ─── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Performance Chart (spans 2 cols) */}
        <div className="lg:col-span-2">
          <PerformanceChart assets={dashboardData} />
              </div>

        {/* Recent News */}
        <div className="lg:col-span-1">
          <RecentNewsMini />
                    </div>
                  </div>

      {/* ─── Second Row: Radar + Health + Allocation ─── */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
        <OpportunityRadar assets={dashboardData} />
        <PortfolioHealth
          avgScore={metrics.avgScore}
          strongBuyCount={metrics.strongBuyCount}
          favorableCount={metrics.favorableCount}
          totalCount={metrics.totalAssets}
        />
        <AllocationDonut assets={dashboardData} />
                  </div>

      {/* ─── Smart Alerts ─── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <SmartAlerts assets={dashboardData} />

        {/* Quick Glance Table */}
        <QuickGlance assets={dashboardData} navigate={navigate} />
      </div>

      {/* ─── About DCA Scoring Banner ─── */}
      <div className="glass-effect rounded-sm p-5 border-l-4 border-primary" data-testid="info-banner">
        <div className="flex gap-4">
          <AlertCircle className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-bold mb-1 text-sm">ABOUT DCA SCORING</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Our composite score (0-100) combines technical momentum, volatility opportunities, statistical deviations, and macro factors.
              Higher scores indicate better dollar-cost averaging opportunities. Scores above 60 suggest favorable accumulation zones.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
