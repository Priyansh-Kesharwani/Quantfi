import React, { useState, useEffect } from 'react';
import api from '../api';
import { RefreshCw, TrendingUp, AlertCircle } from 'lucide-react';
import { toast } from 'sonner';

const Dashboard = () => {
  const [dashboardData, setDashboardData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchDashboard = async () => {
    try {
      const response = await api.getDashboard();
      setDashboardData(response.data.assets || []);
    } catch (error) {
      console.error('Error fetching dashboard:', error);
      toast.error('Failed to load dashboard data');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchDashboard();
  }, []);

  const handleRefresh = () => {
    setRefreshing(true);
    fetchDashboard();
  };
  
  const getScoreColor = (score) => {
    if (score >= 81) return '#22C55E';
    if (score >= 61) return '#10B981';
    if (score >= 31) return '#F59E0B';
    return '#EF4444';
  };
  
  const getZoneLabel = (zone) => {
    const labels = {
      'strong_buy': 'STRONG BUY DIP',
      'favorable': 'FAVORABLE',
      'neutral': 'NEUTRAL',
      'unfavorable': 'UNFAVORABLE'
    };
    return labels[zone] || (zone || '').toUpperCase();
  };

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

  if (dashboardData.length === 0) {
    return (
      <div className="flex items-center justify-center h-screen" data-testid="empty-dashboard">
        <div className="text-center max-w-md">
          <TrendingUp className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
          <h2 className="text-2xl font-bold mb-2">NO ASSETS IN WATCHLIST</h2>
          <p className="text-muted-foreground mb-6">
            Add assets like GOLD, SILVER, or US equities to start analyzing DCA opportunities.
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
    <div className="p-8" data-testid="dashboard-main">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold tracking-tight mb-2" data-testid="dashboard-title">PORTFOLIO OVERVIEW</h1>
          <p className="text-muted-foreground">Real-time DCA intelligence for your watchlist</p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="flex items-center gap-2 px-4 py-2 glass-effect hover:bg-white/10 rounded transition"
          data-testid="refresh-dashboard-btn"
        >
          <RefreshCw className={`w-5 h-5 ${refreshing ? 'animate-spin' : ''}`} />
          REFRESH
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6" data-testid="asset-grid">
        {dashboardData.map((item) => {
          const asset = item.asset;
          const score = item.score;
          const price = item.price;
          const indicators = item.indicators;
          
          const scoreValue = score?.composite_score || 0;
          const zone = score?.zone || 'neutral';
          const priceUSD = price?.price_usd || 0;
          const priceINR = price?.price_inr || 0;
          const rsi = indicators?.rsi_14;
          const drawdown = indicators?.drawdown_pct;
          
          return (
            <div key={asset.symbol} className="glass-effect rounded-sm p-6 hover:border-primary/30 transition-all h-full" data-testid={`asset-card-${asset.symbol}`}>
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-xl font-bold tracking-tight" data-testid={`asset-symbol-${asset.symbol}`}>
                    {asset.symbol}
                  </h3>
                  <p className="text-sm text-muted-foreground">{asset.name}</p>
                </div>
                <div className={`px-3 py-1 rounded text-xs font-bold score-zone-${zone}`} data-testid={`asset-zone-${asset.symbol}`}>
                  {getZoneLabel(zone)}
                </div>
              </div>
              
              <div className="mb-4">
                <div className="flex items-baseline gap-3">
                  <span className="text-3xl font-data font-semibold" data-testid={`asset-price-usd-${asset.symbol}`}>
                    ${priceUSD.toFixed(2)}
                  </span>
                  <span className="text-lg text-muted-foreground font-data" data-testid={`asset-price-inr-${asset.symbol}`}>
                    ₹{priceINR.toFixed(2)}
                  </span>
                </div>
              </div>
              
              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-muted-foreground">DCA SCORE</span>
                  <span className="text-2xl font-bold font-data" style={{ color: getScoreColor(scoreValue) }} data-testid={`asset-score-${asset.symbol}`}>
                    {scoreValue.toFixed(0)}
                  </span>
                </div>
                <div className="w-full bg-white/5 rounded-full h-2 overflow-hidden">
                  <div
                    className="h-full transition-all duration-300"
                    style={{ width: `${scoreValue}%`, backgroundColor: getScoreColor(scoreValue) }}
                  />
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-3 text-sm">
                {rsi !== null && rsi !== undefined && (
                  <div data-testid={`asset-rsi-${asset.symbol}`}>
                    <div className="text-muted-foreground text-xs">RSI</div>
                    <div className="font-data font-semibold">{rsi.toFixed(1)}</div>
                  </div>
                )}
                {drawdown !== null && drawdown !== undefined && (
                  <div data-testid={`asset-drawdown-${asset.symbol}`}>
                    <div className="text-muted-foreground text-xs">DRAWDOWN</div>
                    <div className="font-data font-semibold text-destructive">{drawdown.toFixed(1)}%</div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-8 glass-effect rounded-sm p-6 border-l-4 border-primary" data-testid="info-banner">
        <div className="flex gap-4">
          <AlertCircle className="w-6 h-6 text-primary flex-shrink-0" />
          <div>
            <h3 className="font-bold mb-1">ABOUT DCA SCORING</h3>
            <p className="text-sm text-muted-foreground">
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
