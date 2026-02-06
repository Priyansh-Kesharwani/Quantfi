import React, { useState, useEffect } from 'react';
import api from '../api';
import { RefreshCw, ExternalLink, TrendingUp, AlertTriangle, Globe } from 'lucide-react';
import { toast } from 'sonner';

const News = () => {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    fetchNews();
  }, []);

  const fetchNews = async () => {
    try {
      const response = await api.getNews();
      setNews(response.data.news || []);
    } catch (error) {
      console.error('Error fetching news:', error);
      toast.error('Failed to load news');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await api.refreshNews();
      await fetchNews();
      toast.success('News refreshed');
    } catch (error) {
      toast.error('Failed to refresh news');
    }
  };

  const getEventTypeIcon = (eventType) => {
    switch (eventType) {
      case 'rate_change':
        return <TrendingUp className="w-5 h-5" />;
      case 'sanction':
      case 'war':
        return <AlertTriangle className="w-5 h-5" />;
      default:
        return <Globe className="w-5 h-5" />;
    }
  };

  const getEventTypeColor = (eventType) => {
    switch (eventType) {
      case 'rate_change':
        return 'text-primary';
      case 'sanction':
      case 'war':
        return 'text-destructive';
      case 'trade_restriction':
        return 'text-chart-5';
      case 'election':
        return 'text-chart-3';
      default:
        return 'text-muted-foreground';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <RefreshCw className="w-12 h-12 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Loading news...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-8" data-testid="news-page">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold tracking-tight mb-2" data-testid="news-title">NEWS & GEOPOLITICAL EVENTS</h1>
          <p className="text-muted-foreground">AI-classified financial news with asset impact analysis</p>
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

      {/* News Feed */}
      {news.length === 0 ? (
        <div className="glass-effect rounded-sm p-12 text-center" data-testid="news-empty">
          <Globe className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
          <h3 className="text-xl font-bold mb-2">NO NEWS AVAILABLE</h3>
          <p className="text-muted-foreground mb-4">
            News data requires a NewsAPI key. Add your key in the backend .env file.
          </p>
        </div>
      ) : (
        <div className="space-y-4" data-testid="news-feed">
          {news.map((article, index) => (
            <div key={article.id || index} className="glass-effect rounded-sm p-6 hover:border-primary/30 transition" data-testid={`news-article-${index}`}>
              <div className="flex gap-4">
                <div className={`flex-shrink-0 ${getEventTypeColor(article.event_type)}`}>
                  {getEventTypeIcon(article.event_type)}
                </div>
                
                <div className="flex-1">
                  <div className="flex items-start justify-between gap-4 mb-2">
                    <h3 className="text-lg font-bold leading-tight" data-testid={`news-title-${index}`}>
                      {article.title}
                    </h3>
                    <span className="text-xs text-muted-foreground whitespace-nowrap">
                      {new Date(article.published_at).toLocaleDateString()}
                    </span>
                  </div>
                  
                  <p className="text-sm text-muted-foreground mb-3">
                    {article.summary || article.description}
                  </p>
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <span className="text-xs px-2 py-1 glass-effect rounded">
                        {article.event_type.replace('_', ' ').toUpperCase()}
                      </span>
                      
                      {article.affected_assets && article.affected_assets.length > 0 && (
                        <div className="flex gap-2">
                          {article.affected_assets.slice(0, 3).map(asset => (
                            <span key={asset} className="text-xs px-2 py-1 bg-primary/10 text-primary rounded font-data">
                              {asset}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                    
                    <a
                      href={article.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1 text-sm text-primary hover:underline"
                      data-testid={`news-link-${index}`}
                    >
                      READ MORE
                      <ExternalLink className="w-4 h-4" />
                    </a>
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
