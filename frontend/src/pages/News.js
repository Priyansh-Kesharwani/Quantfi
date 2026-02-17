import React, { useState, useEffect } from 'react';
import api from '../api';
import { RefreshCw, ExternalLink, TrendingUp, AlertTriangle, Globe, Filter } from 'lucide-react';
import { toast } from 'sonner';

const News = () => {
  const [news, setNews] = useState([]);
  const [assets, setAssets] = useState([]);
  const [filter, setFilter] = useState('ALL');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    const load = async () => {
      try {
        const [newsRes, assetsRes] = await Promise.allSettled([
          api.getNews(50),
          api.getAssets(),
        ]);
        if (newsRes.status === 'fulfilled') setNews(newsRes.value.data.news || []);
        if (assetsRes.status === 'fulfilled') setAssets(assetsRes.value.data || []);
      } catch (e) {
        toast.error('Failed to load news');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await api.refreshNews();
      const res = await api.getNews(50);
      setNews(res.data.news || []);
      toast.success('News refreshed');
    } catch (e) {
      toast.error('Failed to refresh news');
    } finally {
      setRefreshing(false);
    }
  };

  const filtered = filter === 'ALL'
    ? news
    : news.filter(a =>
        (a.affected_assets || []).includes(filter) ||
        (a.title || '').toUpperCase().includes(filter) ||
        (a.description || '').toUpperCase().includes(filter)
      );

  const getEventIcon = (type) => {
    switch (type) {
      case 'rate_change': return <TrendingUp className="w-4 h-4" />;
      case 'sanction': case 'war': return <AlertTriangle className="w-4 h-4" />;
      default: return <Globe className="w-4 h-4" />;
    }
  };

  const formatAge = (dt) => {
    if (!dt) return '';
    const h = Math.floor((Date.now() - new Date(dt).getTime()) / 3600000);
    if (h < 1) return 'Just now';
    if (h < 24) return `${h}h ago`;
    return `${Math.floor(h / 24)}d ago`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <RefreshCw className="w-12 h-12 animate-spin mx-auto mb-4 text-primary" />
      </div>
    );
  }

  return (
    <div className="p-8" data-testid="news-page">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-4xl font-bold tracking-tight mb-2" data-testid="news-title">NEWS & GEOPOLITICAL EVENTS</h1>
          <p className="text-muted-foreground text-sm">Asset-relevant news from your watchlist</p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="flex items-center gap-2 px-4 py-2 glass-effect hover:bg-white/10 rounded transition"
          data-testid="refresh-news-btn"
        >
          <RefreshCw className={`w-5 h-5 ${refreshing ? 'animate-spin' : ''}`} />
          REFRESH
        </button>
      </div>

      {/* Asset Filter Tabs */}
      <div className="flex items-center gap-2 mb-6 flex-wrap" data-testid="news-filter-tabs">
        <Filter className="w-4 h-4 text-muted-foreground" />
        <button
          onClick={() => setFilter('ALL')}
          className={`px-3 py-1 rounded text-xs font-bold transition ${filter === 'ALL' ? 'bg-primary text-primary-foreground' : 'glass-effect hover:bg-white/10'}`}
        >
          ALL
        </button>
        {assets.map(a => (
          <button
            key={a.symbol}
            onClick={() => setFilter(a.symbol)}
            className={`px-3 py-1 rounded text-xs font-bold transition ${filter === a.symbol ? 'bg-primary text-primary-foreground' : 'glass-effect hover:bg-white/10'}`}
          >
            {a.symbol}
          </button>
        ))}
      </div>

      {filtered.length === 0 ? (
        <div className="glass-effect rounded-sm p-12 text-center" data-testid="news-empty">
          <Globe className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
          <h3 className="text-xl font-bold mb-2">NO NEWS AVAILABLE</h3>
          <p className="text-muted-foreground mb-4">
            {filter !== 'ALL'
              ? `No news found for ${filter}. Try selecting ALL or click REFRESH.`
              : 'Click REFRESH to fetch the latest financial headlines.'}
          </p>
        </div>
      ) : (
        <div className="space-y-3" data-testid="news-feed">
          {filtered.map((article, index) => (
            <div key={article.id || index} className="glass-effect rounded-sm p-5 hover:border-primary/30 transition" data-testid={`news-article-${index}`}>
              <div className="flex gap-3">
                <div className="flex-shrink-0 text-muted-foreground mt-0.5">
                  {getEventIcon(article.event_type)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-3 mb-1">
                    <h3 className="text-sm font-bold leading-tight" data-testid={`news-title-${index}`}>
                      {article.title}
                    </h3>
                    <span className="text-[10px] text-muted-foreground whitespace-nowrap shrink-0">
                      {formatAge(article.published_at)}
                    </span>
                  </div>

                  <p className="text-xs text-muted-foreground mb-2 line-clamp-2">
                    {article.summary || article.description}
                  </p>

                  <div className="flex items-center justify-between flex-wrap gap-2">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-[10px] px-1.5 py-0.5 glass-effect rounded font-bold">
                        {(article.event_type || 'general').replace('_', ' ').toUpperCase()}
                      </span>
                      {article.source && (
                        <span className="text-[10px] text-muted-foreground">{article.source}</span>
                      )}
                      {(article.affected_assets || []).map(asset => (
                        <span key={asset} className="text-[10px] px-1.5 py-0.5 bg-primary/10 text-primary rounded font-data font-bold">
                          {asset}
                        </span>
                      ))}
                    </div>
                    {article.url && (
                      <a
                        href={article.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-1 text-[10px] text-primary hover:underline shrink-0"
                        data-testid={`news-link-${index}`}
                      >
                        READ MORE <ExternalLink className="w-3 h-3" />
                      </a>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default News;
