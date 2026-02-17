import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api';
import { formatCurrency, getScoreColor, getZoneLabel } from '../utils';
import {
  RefreshCw, TrendingUp, Search, Filter, LayoutGrid,
  LayoutList, ChevronDown, ChevronUp, AlertTriangle, Plus,
  Trash2, ExternalLink, ArrowUpDown, X
} from 'lucide-react';
import { toast } from 'sonner';
import AssetNewsWidget from '../components/AssetNewsWidget';

/* ─── Asset Category Tabs ─── */
const CATEGORIES = [
  { key: 'all', label: 'ALL' },
  { key: 'equity', label: 'EQUITY' },
  { key: 'etf', label: 'ETF' },
  { key: 'commodity', label: 'COMMODITY' },
  { key: 'crypto', label: 'CRYPTO' },
  { key: 'index', label: 'INDEX' },
];

/* ─── Sort Options ─── */
const SORT_OPTIONS = [
  { key: 'score_desc', label: 'Score (High → Low)' },
  { key: 'score_asc', label: 'Score (Low → High)' },
  { key: 'name_asc', label: 'Name (A → Z)' },
  { key: 'name_desc', label: 'Name (Z → A)' },
  { key: 'price_desc', label: 'Price (High → Low)' },
  { key: 'price_asc', label: 'Price (Low → High)' },
  { key: 'rsi_asc', label: 'RSI (Low → High)' },
  { key: 'drawdown_asc', label: 'Drawdown (Worst first)' },
];

/* ─── Zone Filter ─── */
const ZONE_FILTERS = [
  { key: 'all', label: 'All Zones', color: '' },
  { key: 'strong_buy', label: 'Strong Buy', color: 'bg-emerald-500/20 text-emerald-400' },
  { key: 'favorable', label: 'Favorable', color: 'bg-teal-500/20 text-teal-400' },
  { key: 'neutral', label: 'Neutral', color: 'bg-amber-500/20 text-amber-400' },
  { key: 'unfavorable', label: 'Unfavorable', color: 'bg-red-500/20 text-red-400' },
];

/* ─── Score Heatmap Bar (Novel) ─── */
const ScoreHeatmap = ({ assets }) => {
  const sorted = useMemo(() => {
    return [...assets]
      .filter(item => item.score?.composite_score != null)
      .sort((a, b) => (b.score?.composite_score || 0) - (a.score?.composite_score || 0));
  }, [assets]);

  if (sorted.length === 0) return null;

  return (
    <div className="glass-effect rounded-sm p-4 mb-6">
      <div className="flex items-center gap-2 mb-3">
        <div className="text-xs text-muted-foreground uppercase tracking-wider">Score Heatmap</div>
        <div className="flex-1" />
        <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
          <span className="flex items-center gap-1"><span className="w-3 h-1.5 rounded bg-red-500/80" /> Low</span>
          <span className="flex items-center gap-1"><span className="w-3 h-1.5 rounded bg-amber-500/80" /> Mid</span>
          <span className="flex items-center gap-1"><span className="w-3 h-1.5 rounded bg-emerald-500/80" /> High</span>
        </div>
      </div>
      <div className="flex gap-1 h-8">
        {sorted.map(item => {
          const score = item.score.composite_score;
          return (
            <div
              key={item.asset.symbol}
              className="flex-1 rounded-sm relative group cursor-pointer transition-all hover:scale-y-125"
              style={{ backgroundColor: getScoreColor(score), opacity: 0.7 + (score / 333) }}
              title={`${item.asset.symbol}: ${score.toFixed(0)}`}
            >
              <div className="absolute -top-8 left-1/2 -translate-x-1/2 hidden group-hover:block glass-effect px-2 py-1 rounded text-[10px] font-data font-bold whitespace-nowrap z-10 border border-white/10">
                {item.asset.symbol}: {score.toFixed(0)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

/* ─── Asset Card (Grid View) ─── */
const AssetGridCard = ({ item, onNavigate, onRemove }) => {
  const asset = item.asset;
  const score = item.score;
  const price = item.price;
  const indicators = item.indicators;

  const hasData = !!(price?.price_usd || score?.composite_score);
  const scoreValue = score?.composite_score ?? 0;
  const zone = score?.zone || (hasData ? 'neutral' : undefined);

  return (
    <div
      className="glass-effect rounded-sm p-5 hover:border-primary/30 transition-all cursor-pointer group relative"
      onClick={() => onNavigate(`/assets/${asset.symbol}`)}
      data-testid={`asset-card-${asset.symbol}`}
    >
      {/* Remove button */}
      <button
        onClick={(e) => { e.stopPropagation(); onRemove(asset.symbol); }}
        className="absolute top-3 right-3 p-1.5 rounded opacity-0 group-hover:opacity-100 hover:bg-destructive/20 text-muted-foreground hover:text-destructive transition"
        title="Remove asset"
      >
        <Trash2 className="w-3.5 h-3.5" />
      </button>

      <div className="flex items-start justify-between mb-3">
        <div className="min-w-0 flex-1 mr-3">
          <h3 className="text-lg font-bold tracking-tight truncate">{asset.symbol}</h3>
          <p className="text-xs text-muted-foreground truncate">{asset.name}</p>
          <span className="text-[10px] text-muted-foreground/60">{(asset.asset_type || '').toUpperCase()}</span>
        </div>
        <div className={`px-2.5 py-1 rounded text-[10px] font-bold whitespace-nowrap score-zone-${zone || 'neutral'}`}>
          {getZoneLabel(zone)}
        </div>
      </div>

      {!hasData ? (
        <div className="flex items-center gap-2 p-2.5 rounded bg-destructive/10 text-destructive/80 text-xs mb-3">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>Data unavailable — try refreshing.</span>
        </div>
      ) : (
        <>
          <div className="mb-3">
            <div className="flex items-baseline gap-2 flex-wrap">
              <span className="text-xl font-data font-semibold whitespace-nowrap">
                {formatCurrency(price?.price_usd, 'USD')}
              </span>
              {price?.price_inr != null && (
                <span className="text-sm text-muted-foreground font-data whitespace-nowrap">
                  {formatCurrency(price?.price_inr, 'INR')}
                </span>
              )}
            </div>
          </div>

          <div className="mb-3">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs text-muted-foreground">DCA SCORE</span>
              <span className="text-xl font-bold font-data" style={{ color: getScoreColor(scoreValue) }}>
                {scoreValue.toFixed(0)}
              </span>
            </div>
            <div className="w-full bg-white/5 rounded-full h-1.5 overflow-hidden">
              <div
                className="h-full transition-all duration-300"
                style={{ width: `${scoreValue}%`, backgroundColor: getScoreColor(scoreValue) }}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 text-sm">
            {indicators?.rsi_14 != null && (
              <div>
                <div className="text-muted-foreground text-[10px]">RSI</div>
                <div className="font-data font-semibold text-xs">{indicators.rsi_14.toFixed(1)}</div>
              </div>
            )}
            {indicators?.drawdown_pct != null && (
              <div>
                <div className="text-muted-foreground text-[10px]">DRAWDOWN</div>
                <div className="font-data font-semibold text-xs text-destructive">{indicators.drawdown_pct.toFixed(1)}%</div>
              </div>
            )}
          </div>
        </>
      )}

      <AssetNewsWidget symbol={asset.symbol} compact={true} limit={1} />
    </div>
  );
};

/* ─── Asset Table Row (List View) ─── */
const AssetTableRow = ({ item, onNavigate, onRemove }) => {
  const asset = item.asset;
  const score = item.score;
  const price = item.price;
  const indicators = item.indicators;
  const scoreValue = score?.composite_score ?? 0;
  const zone = score?.zone || 'neutral';

  return (
    <tr
      className="border-b border-white/[0.04] hover:bg-white/[0.03] cursor-pointer transition group"
      onClick={() => onNavigate(`/assets/${asset.symbol}`)}
      data-testid={`asset-row-${asset.symbol}`}
    >
      <td className="py-4 pl-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-white/5 flex items-center justify-center text-xs font-bold font-data text-primary">
            {asset.symbol.slice(0, 2)}
          </div>
          <div>
            <div className="text-sm font-bold">{asset.symbol}</div>
            <div className="text-xs text-muted-foreground">{asset.name}</div>
          </div>
        </div>
      </td>
      <td className="py-4 text-right">
        <span className="text-[10px] px-2 py-0.5 rounded font-bold bg-white/5 text-muted-foreground uppercase">
          {asset.asset_type}
        </span>
      </td>
      <td className="py-4 text-right">
        <div className="text-sm font-data">{formatCurrency(price?.price_usd, 'USD')}</div>
        {price?.price_inr != null && (
          <div className="text-xs text-muted-foreground font-data">{formatCurrency(price?.price_inr, 'INR')}</div>
        )}
      </td>
      <td className="py-4 text-right">
        <div className="flex items-center justify-end gap-2">
          <div className="w-16 bg-white/5 rounded-full h-1.5 overflow-hidden">
            <div className="h-full" style={{ width: `${scoreValue}%`, backgroundColor: getScoreColor(scoreValue) }} />
          </div>
          <span className="text-sm font-data font-bold w-8 text-right" style={{ color: getScoreColor(scoreValue) }}>
            {scoreValue.toFixed(0)}
          </span>
        </div>
      </td>
      <td className="py-4 text-right">
        <span className={`text-[10px] px-2 py-0.5 rounded font-bold score-zone-${zone}`}>
          {getZoneLabel(zone)}
        </span>
      </td>
      <td className="py-4 text-right font-data text-sm">
        {indicators?.rsi_14 != null ? indicators.rsi_14.toFixed(1) : '—'}
      </td>
      <td className="py-4 text-right font-data text-sm">
        {indicators?.drawdown_pct != null ? (
          <span className={indicators.drawdown_pct < -10 ? 'text-red-400' : ''}>
            {indicators.drawdown_pct.toFixed(1)}%
          </span>
        ) : '—'}
      </td>
      <td className="py-4 pr-4 text-right">
        <div className="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition">
          <button
            onClick={(e) => { e.stopPropagation(); onNavigate(`/assets/${asset.symbol}`); }}
            className="p-1.5 rounded hover:bg-white/10 text-muted-foreground hover:text-primary transition"
            title="View details"
          >
            <ExternalLink className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onRemove(asset.symbol); }}
            className="p-1.5 rounded hover:bg-destructive/20 text-muted-foreground hover:text-destructive transition"
            title="Remove"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </td>
    </tr>
  );
};

/* ──────────────────────────────
   MAIN ASSETS PAGE
   ────────────────────────────── */
const Assets = ({ refreshKey = 0 }) => {
  const navigate = useNavigate();
  const [dashboardData, setDashboardData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  /* ─── View State ─── */
  const [viewMode, setViewMode] = useState('grid'); // 'grid' | 'list'
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [zoneFilter, setZoneFilter] = useState('all');
  const [sortBy, setSortBy] = useState('score_desc');
  const [showSortDropdown, setShowSortDropdown] = useState(false);

  const fetchDashboard = async () => {
    try {
      const response = await api.getDashboard();
      const assets = response.data.assets || [];
      setDashboardData(assets);
    } catch (error) {
      console.error('[Assets] Error fetching:', error);
      toast.error('Failed to load assets data');
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

  const handleRemoveAsset = async (symbol) => {
    if (!window.confirm(`Remove ${symbol} from watchlist?`)) return;
    try {
      await api.removeAsset(symbol);
      toast.success(`${symbol} removed from watchlist`);
      setDashboardData(prev => prev.filter(item => item.asset.symbol !== symbol));
    } catch (err) {
      toast.error('Failed to remove asset');
    }
  };

  /* ─── Filter + Sort ─── */
  const filteredAssets = useMemo(() => {
    let result = [...dashboardData];

    // Search
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(item =>
        item.asset.symbol.toLowerCase().includes(q) ||
        (item.asset.name || '').toLowerCase().includes(q)
      );
    }

    // Category filter
    if (categoryFilter !== 'all') {
      result = result.filter(item => item.asset.asset_type === categoryFilter);
    }

    // Zone filter
    if (zoneFilter !== 'all') {
      result = result.filter(item => item.score?.zone === zoneFilter);
    }

    // Sort
    const [key, dir] = sortBy.split('_');
    result.sort((a, b) => {
      let aVal, bVal;
      switch (key) {
        case 'score':
          aVal = a.score?.composite_score ?? 0;
          bVal = b.score?.composite_score ?? 0;
          break;
        case 'name':
          aVal = a.asset.symbol;
          bVal = b.asset.symbol;
          return dir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
        case 'price':
          aVal = a.price?.price_usd ?? 0;
          bVal = b.price?.price_usd ?? 0;
          break;
        case 'rsi':
          aVal = a.indicators?.rsi_14 ?? 999;
          bVal = b.indicators?.rsi_14 ?? 999;
          break;
        case 'drawdown':
          aVal = a.indicators?.drawdown_pct ?? 0;
          bVal = b.indicators?.drawdown_pct ?? 0;
          break;
        default:
          aVal = 0;
          bVal = 0;
      }
      return dir === 'asc' ? aVal - bVal : bVal - aVal;
    });

    return result;
  }, [dashboardData, searchQuery, categoryFilter, zoneFilter, sortBy]);

  const activeFiltersCount = [
    categoryFilter !== 'all' ? 1 : 0,
    zoneFilter !== 'all' ? 1 : 0,
    searchQuery.trim() ? 1 : 0,
  ].reduce((a, b) => a + b, 0);

  /* ─── Loading ─── */
  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen" data-testid="loading-assets">
        <div className="text-center">
          <RefreshCw className="w-12 h-12 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Loading assets...</p>
        </div>
      </div>
    );
  }

  /* ─── Empty State ─── */
  if (dashboardData.length === 0) {
    return (
      <div className="flex items-center justify-center h-screen" data-testid="empty-assets">
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
    <div className="p-6 lg:p-8 max-w-[1600px] mx-auto" data-testid="assets-page">
      {/* ─── Header ─── */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-4xl font-bold tracking-tight mb-1" data-testid="assets-title">ASSETS</h1>
          <p className="text-muted-foreground text-sm">
            {dashboardData.length} asset{dashboardData.length !== 1 ? 's' : ''} in watchlist
            {filteredAssets.length !== dashboardData.length && (
              <span className="text-primary ml-1">· {filteredAssets.length} shown</span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => window.dispatchEvent(new Event('addAsset'))}
            className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded font-medium text-sm hover:bg-primary/90 transition"
          >
            <Plus className="w-4 h-4" />
            ADD ASSET
          </button>
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="flex items-center gap-2 px-4 py-2 glass-effect hover:bg-white/10 rounded transition"
            data-testid="refresh-assets-btn"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            <span className="text-sm">REFRESH</span>
          </button>
        </div>
      </div>

      {/* ─── Score Heatmap (Novel) ─── */}
      <ScoreHeatmap assets={dashboardData} />

      {/* ─── Toolbar: Search + Filters + Sort + View Toggle ─── */}
      <div className="flex flex-col lg:flex-row items-start lg:items-center gap-3 mb-6">
        {/* Search */}
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search assets..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-10 py-2.5 glass-effect rounded text-sm font-data focus:outline-none focus:ring-1 focus:ring-primary/50"
            data-testid="asset-search-input"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Category Tabs */}
        <div className="flex items-center gap-1 flex-wrap">
          {CATEGORIES.map(cat => (
            <button
              key={cat.key}
              onClick={() => setCategoryFilter(cat.key)}
              className={`px-3 py-1.5 rounded text-xs font-bold transition ${
                categoryFilter === cat.key
                  ? 'bg-primary text-primary-foreground'
                  : 'glass-effect hover:bg-white/10'
              }`}
            >
              {cat.label}
            </button>
          ))}
        </div>

        {/* Zone Filter */}
        <div className="flex items-center gap-1">
          <Filter className="w-4 h-4 text-muted-foreground mr-1" />
          {ZONE_FILTERS.map(z => (
            <button
              key={z.key}
              onClick={() => setZoneFilter(z.key)}
              className={`px-2.5 py-1 rounded text-[10px] font-bold transition ${
                zoneFilter === z.key
                  ? z.key === 'all' ? 'bg-primary text-primary-foreground' : z.color + ' ring-1 ring-current'
                  : z.key === 'all' ? 'glass-effect hover:bg-white/10' : 'glass-effect hover:bg-white/10 text-muted-foreground'
              }`}
            >
              {z.label}
            </button>
          ))}
        </div>

        {/* Sort + View Toggle */}
        <div className="flex items-center gap-2 ml-auto">
          {/* Sort Dropdown */}
          <div className="relative">
            <button
              onClick={() => setShowSortDropdown(!showSortDropdown)}
              className="flex items-center gap-2 px-3 py-2 glass-effect rounded text-xs hover:bg-white/10 transition"
            >
              <ArrowUpDown className="w-3.5 h-3.5" />
              SORT
              {showSortDropdown ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>
            {showSortDropdown && (
              <div className="absolute right-0 top-full mt-1 w-48 glass-effect rounded shadow-xl border border-white/10 z-20 py-1">
                {SORT_OPTIONS.map(opt => (
                  <button
                    key={opt.key}
                    onClick={() => { setSortBy(opt.key); setShowSortDropdown(false); }}
                    className={`w-full text-left px-3 py-2 text-xs hover:bg-white/10 transition ${
                      sortBy === opt.key ? 'text-primary font-bold' : ''
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* View Toggle */}
          <div className="flex items-center glass-effect rounded overflow-hidden">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 transition ${viewMode === 'grid' ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:text-foreground'}`}
              title="Grid view"
            >
              <LayoutGrid className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 transition ${viewMode === 'list' ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:text-foreground'}`}
              title="List view"
            >
              <LayoutList className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Active Filters Badge */}
      {activeFiltersCount > 0 && (
        <div className="flex items-center gap-2 mb-4">
          <span className="text-xs text-muted-foreground">{activeFiltersCount} filter{activeFiltersCount > 1 ? 's' : ''} active</span>
          <button
            onClick={() => {
              setSearchQuery('');
              setCategoryFilter('all');
              setZoneFilter('all');
            }}
            className="text-xs text-primary hover:underline"
          >
            Clear all
          </button>
        </div>
      )}

      {/* ─── No Results ─── */}
      {filteredAssets.length === 0 && (
        <div className="glass-effect rounded-sm p-12 text-center">
          <Search className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
          <h3 className="text-xl font-bold mb-2">NO MATCHING ASSETS</h3>
          <p className="text-muted-foreground text-sm">
            Try adjusting your search or filters to find assets.
          </p>
        </div>
      )}

      {/* ─── Grid View ─── */}
      {viewMode === 'grid' && filteredAssets.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5" data-testid="asset-grid">
          {filteredAssets.map(item => (
            <AssetGridCard
              key={item.asset.symbol}
              item={item}
              onNavigate={navigate}
              onRemove={handleRemoveAsset}
            />
          ))}
        </div>
      )}

      {/* ─── List/Table View ─── */}
      {viewMode === 'list' && filteredAssets.length > 0 && (
        <div className="glass-effect rounded-sm overflow-hidden" data-testid="asset-table">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left text-[10px] text-muted-foreground py-3 pl-4 uppercase tracking-wider">Asset</th>
                  <th className="text-right text-[10px] text-muted-foreground py-3 uppercase tracking-wider">Type</th>
                  <th className="text-right text-[10px] text-muted-foreground py-3 uppercase tracking-wider">Price</th>
                  <th className="text-right text-[10px] text-muted-foreground py-3 uppercase tracking-wider">DCA Score</th>
                  <th className="text-right text-[10px] text-muted-foreground py-3 uppercase tracking-wider">Zone</th>
                  <th className="text-right text-[10px] text-muted-foreground py-3 uppercase tracking-wider">RSI</th>
                  <th className="text-right text-[10px] text-muted-foreground py-3 uppercase tracking-wider">Drawdown</th>
                  <th className="text-right text-[10px] text-muted-foreground py-3 pr-4 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredAssets.map(item => (
                  <AssetTableRow
                    key={item.asset.symbol}
                    item={item}
                    onNavigate={navigate}
                    onRemove={handleRemoveAsset}
                  />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default Assets;
