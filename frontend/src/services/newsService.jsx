import api from '@/api';

function normalizeArticle(article) {
  let sentiment = 'neutral';
  if (article.impact_scores && typeof article.impact_scores === 'object') {
    const overall = article.impact_scores.overall ?? article.impact_scores.confidence ?? null;
    if (overall !== null) {
      sentiment = overall > 0.6 ? 'positive' : overall < 0.4 ? 'negative' : 'neutral';
    }
  }
  if (article.event_type === 'rate_change' || article.event_type === 'war') {
    sentiment = 'negative';
  }

  return {
    title: article.title || 'Untitled',
    source: article.source || '',
    timestamp: article.published_at || '',
    sentiment,
  };
}

export async function getNewsForAsset(ticker, limit = 2) {
  const symbol = (ticker || '').toUpperCase();

  try {
    const response = await api.getNewsForAsset(symbol, limit);
    const articles = response?.data?.news;

    if (Array.isArray(articles) && articles.length > 0) {
      return articles.slice(0, limit).map(normalizeArticle);
    }
  } catch (err) {
    console.warn(`[newsService] Backend error for ${symbol}: ${err.message}`);
  }

  return [];
}

export function getSentimentColor(sentiment) {
  switch (sentiment) {
    case 'positive':
      return 'text-emerald-400 bg-emerald-400/10';
    case 'negative':
      return 'text-red-400 bg-red-400/10';
    default:
      return 'text-zinc-400 bg-zinc-400/10';
  }
}

export function formatRelativeTime(isoTimestamp) {
  if (!isoTimestamp) return '';
  const diffMs = Date.now() - new Date(isoTimestamp).getTime();
  const diffH = Math.floor(diffMs / (1000 * 60 * 60));
  if (diffH < 1) return 'Just now';
  if (diffH < 24) return `${diffH}h ago`;
  const diffD = Math.floor(diffH / 24);
  return `${diffD}d ago`;
}
