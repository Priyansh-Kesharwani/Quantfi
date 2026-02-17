import React, { useState, useEffect } from 'react';
import { Newspaper } from 'lucide-react';
import { getNewsForAsset, getSentimentColor, formatRelativeTime } from '../services/newsService';

/**
 * Asset News Widget — shows top 1-2 headlines for a specific asset.
 * Renders below RSI + Drawdown row on asset cards and detail pages.
 *
 * @param {Object} props
 * @param {string} props.symbol - Asset ticker symbol
 * @param {number} [props.limit=2] - Max headlines
 * @param {boolean} [props.compact=false] - Compact mode (single headline for dashboard cards)
 */
const AssetNewsWidget = ({ symbol, limit = 2, compact = false }) => {
  const [headlines, setHeadlines] = useState([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const fetchNews = async () => {
      try {
        console.debug(`[AssetNewsWidget] Fetching news for ${symbol}`);
        const data = await getNewsForAsset(symbol, limit);
        console.debug(`[AssetNewsWidget] Got ${data?.length || 0} headlines for ${symbol}`);
        if (!cancelled) {
          setHeadlines(data || []);
          setLoaded(true);
        }
      } catch (err) {
        console.warn(`[AssetNewsWidget] Error for ${symbol}:`, err.message);
        if (!cancelled) {
          setHeadlines([]);
          setLoaded(true);
        }
      }
    };
    fetchNews();
    return () => { cancelled = true; };
  }, [symbol, limit]);

  if (!loaded) {
    return (
      <div className="mt-3 pt-3 border-t border-white/5 animate-pulse" data-testid={`news-widget-loading-${symbol}`}>
        <div className="h-3 bg-white/5 rounded w-3/4 mb-1" />
        <div className="h-2 bg-white/5 rounded w-1/2" />
      </div>
    );
  }

  // No headlines available
  if (!headlines.length || (headlines.length === 1 && !headlines[0].source)) {
    return (
      <div className="mt-3 pt-3 border-t border-white/5" data-testid={`news-widget-empty-${symbol}`}>
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Newspaper className="w-3 h-3" />
          <span>No news available</span>
        </div>
      </div>
    );
  }

  const displayHeadlines = compact ? headlines.slice(0, 1) : headlines;

  return (
    <div className="mt-3 pt-3 border-t border-white/5" data-testid={`news-widget-${symbol}`}>
      <div className="flex items-center gap-1.5 mb-2">
        <Newspaper className="w-3 h-3 text-muted-foreground" />
        <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Top News</span>
      </div>
      <div className="space-y-1.5">
        {displayHeadlines.map((item, idx) => (
          <div key={idx} className="group" data-testid={`news-headline-${symbol}-${idx}`}>
            <p className="text-xs leading-tight text-foreground/80 group-hover:text-foreground transition line-clamp-1">
              {item.title}
            </p>
            <div className="flex items-center gap-2 mt-0.5">
              {item.source && (
                <span className="text-[10px] text-muted-foreground">{item.source}</span>
              )}
              {item.timestamp && (
                <span className="text-[10px] text-muted-foreground">{formatRelativeTime(item.timestamp)}</span>
              )}
              <span className={`text-[10px] px-1.5 py-0 rounded-sm font-medium ${getSentimentColor(item.sentiment)}`}>
                {item.sentiment?.toUpperCase()}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AssetNewsWidget;
