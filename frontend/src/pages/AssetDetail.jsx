import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';

import AssetNewsWidget from '@/components/AssetNewsWidget';

import api from '@/api';
import { formatCurrency, formatNumber, formatPercent, getScoreColor, getZoneLabel } from '@/utils';
import { ArrowLeft, Activity, Zap, AlertTriangle, TrendingUp, TrendingDown, Minus, Brain, RefreshCw, ExternalLink, MessageSquare, Newspaper, BookOpen } from 'lucide-react';
import { toast } from 'sonner';

const IndicatorItem = ({ label, value }) => {
  if (value === null || value === undefined) return null;
  return (
    <div>
      <div className="text-xs text-muted-foreground mb-1">{label}</div>
      <div className="font-data font-semibold">{value}</div>
    </div>
  );
};

const IndicatorGrid = ({ indicators }) => {
  if (!indicators) return null;
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-6">
      <IndicatorItem label="SMA 50" value={indicators.sma_50 ? formatCurrency(indicators.sma_50, 'USD') : null} />
      <IndicatorItem label="SMA 200" value={indicators.sma_200 ? formatCurrency(indicators.sma_200, 'USD') : null} />
      <IndicatorItem label="RSI (14)" value={indicators.rsi_14 != null ? formatNumber(indicators.rsi_14, 1) : null} />
      <IndicatorItem label="MACD" value={indicators.macd != null ? formatNumber(indicators.macd, 2) : null} />
      <IndicatorItem label="ATR (14)" value={indicators.atr_14 ? formatNumber(indicators.atr_14, 2) : null} />
      <IndicatorItem label="ADX (14)" value={indicators.adx_14 ? formatNumber(indicators.adx_14, 1) : null} />
      <IndicatorItem label="DRAWDOWN" value={indicators.drawdown_pct != null ? formatPercent(indicators.drawdown_pct, 1) : null} />
      <IndicatorItem label="Z-SCORE (50)" value={indicators.z_score_50 != null ? formatNumber(indicators.z_score_50, 2) : null} />
      <IndicatorItem label="BB UPPER" value={indicators.bb_upper ? formatCurrency(indicators.bb_upper, 'USD') : null} />
      <IndicatorItem label="BB LOWER" value={indicators.bb_lower ? formatCurrency(indicators.bb_lower, 'USD') : null} />
    </div>
  );
};

const DirectionBadge = ({ direction }) => {
  const cfg = {
    bullish: { icon: TrendingUp, color: 'text-emerald-400 bg-emerald-400/10', label: 'BULLISH' },
    bearish: { icon: TrendingDown, color: 'text-red-400 bg-red-400/10', label: 'BEARISH' },
    neutral: { icon: Minus, color: 'text-zinc-400 bg-zinc-400/10', label: 'NEUTRAL' },
  };
  const c = cfg[direction] || cfg.neutral;
  const Icon = c.icon;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold ${c.color}`}>
      <Icon className="w-3 h-3" />
      {c.label}
    </span>
  );
};

const TierBadge = ({ sourceId }) => {
  const tier = sourceId?.charAt(0);
  const cfg = {
    N: { label: 'NEWS', color: 'bg-blue-500/20 text-blue-400', icon: Newspaper },
    R: { label: 'REDDIT', color: 'bg-orange-500/20 text-orange-400', icon: MessageSquare },
    B: { label: 'BLOG', color: 'bg-purple-500/20 text-purple-400', icon: BookOpen },
  };
  const c = cfg[tier] || { label: sourceId, color: 'bg-zinc-500/20 text-zinc-400', icon: Newspaper };
  const Icon = c.icon;
  return (
    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-mono font-bold ${c.color}`}>
      <Icon className="w-2.5 h-2.5" />
      {sourceId}
    </span>
  );
};

const SentimentPanel = ({ symbol }) => {
  const [sentiment, setSentiment] = useState(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    const fetchSentiment = async () => {
      try {
        const res = await api.getSentiment(symbol);
        setSentiment(res.data);
      } catch {
        // Sentiment not available
      } finally {
        setLoading(false);
      }
    };
    fetchSentiment();
  }, [symbol]);

  const handleRun = async () => {
    setRunning(true);
    try {
      const res = await api.runSentiment(symbol);
      setSentiment(res.data);
      toast.success('Sentiment analysis complete');
    } catch {
      toast.error('Sentiment analysis failed');
    } finally {
      setRunning(false);
    }
  };

  const hasResult = sentiment && sentiment.confidence > 0;
  const gValue = sentiment?.G_t ?? 1.0;
  const gColor = gValue > 1.02 ? 'text-emerald-400' : gValue < 0.98 ? 'text-red-400' : 'text-zinc-400';
  const gLabel = gValue > 1.02 ? 'AMPLIFY' : gValue < 0.98 ? 'DAMPEN' : 'NEUTRAL';

  return (
    <div className="glass-effect rounded-sm p-6 mb-6" data-testid="sentiment-panel">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-primary" />
          <h2 className="text-xl font-bold">SENTIMENT ANALYSIS</h2>
        </div>
        <button onClick={handleRun} disabled={running}
          className="flex items-center gap-2 px-3 py-1.5 text-xs glass-effect hover:bg-white/10 rounded transition"
          data-testid="run-sentiment-btn">
          <RefreshCw className={`w-3.5 h-3.5 ${running ? 'animate-spin' : ''}`} />
          {running ? 'ANALYZING...' : 'RUN ANALYSIS'}
        </button>
      </div>

      {loading ? (
        <div className="animate-pulse space-y-3">
          <div className="h-4 bg-white/5 rounded w-1/3" />
          <div className="h-4 bg-white/5 rounded w-2/3" />
        </div>
      ) : !hasResult ? (
        <div className="text-center py-6 text-muted-foreground">
          <Brain className="w-10 h-10 mx-auto mb-2 opacity-30" />
          <p className="text-sm">Click RUN ANALYSIS to generate sentiment from news, Reddit, and expert blogs.</p>
          <p className="text-xs mt-1 opacity-60">Requires an LLM API key (OpenAI/Anthropic) in .env</p>
        </div>
      ) : (
        <div className="space-y-5">
          <div className="flex items-center gap-6">
            <div>
              <div className="text-xs text-muted-foreground mb-1">SENTIMENT GATE (G<sub>t</sub>)</div>
              <div className={`text-4xl font-bold font-data ${gColor}`}>{gValue.toFixed(4)}</div>
              <div className={`text-xs font-bold mt-1 ${gColor}`}>{gLabel}</div>
            </div>
            <div className="flex-1 grid grid-cols-3 gap-4">
              <div>
                <div className="text-[10px] text-muted-foreground">RAW SENTIMENT</div>
                <div className="font-data font-bold">{(sentiment.raw_sentiment ?? 0).toFixed(2)}</div>
              </div>
              <div>
                <div className="text-[10px] text-muted-foreground">CONFIDENCE</div>
                <div className="font-data font-bold">{((sentiment.confidence ?? 0) * 100).toFixed(0)}%</div>
              </div>
              <div>
                <div className="text-[10px] text-muted-foreground">SOURCE AGREEMENT</div>
                <div className="font-data font-bold">{((sentiment.source_agreement ?? 0) * 100).toFixed(0)}%</div>
              </div>
            </div>
          </div>

          {sentiment.reasoning && (
            <div className="text-sm text-muted-foreground leading-relaxed border-l-2 border-primary/30 pl-3">
              {sentiment.reasoning}
            </div>
          )}

          {sentiment.top_factors?.length > 0 && (
            <div>
              <div className="text-xs text-muted-foreground mb-3">TOP FACTORS</div>
              <div className="space-y-2">
                {sentiment.top_factors.map((f, i) => (
                  <div key={i} className="flex items-start gap-3 p-3 rounded bg-white/[0.02] border border-white/5">
                    <div className="text-lg font-bold font-data text-muted-foreground w-6 text-center shrink-0">{f.rank || i + 1}</div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1 flex-wrap">
                        <span className="text-sm font-semibold">{f.factor}</span>
                        <DirectionBadge direction={f.direction} />
                        <span className="text-[10px] text-muted-foreground font-data">impact: {((f.impact_magnitude ?? 0) * 100).toFixed(0)}%</span>
                      </div>
                      {f.explanation && <p className="text-xs text-muted-foreground">{f.explanation}</p>}
                      {f.supporting_sources?.length > 0 && (
                        <div className="flex gap-1.5 mt-1.5 flex-wrap">
                          {f.supporting_sources.map((s, j) => <TierBadge key={j} sourceId={s} />)}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {sentiment.citations?.length > 0 && (
            <div>
              <div className="text-xs text-muted-foreground mb-3">CITATIONS ({sentiment.citations.length})</div>
              <div className="space-y-1.5">
                {sentiment.citations.map((c, i) => (
                  <div key={i} className="flex items-start gap-2 text-xs">
                    <TierBadge sourceId={c.source_id} />
                    <span className="text-muted-foreground flex-1">{c.claim}</span>
                    <DirectionBadge direction={c.direction} />
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-muted-foreground">
            {sentiment.dispersion_analysis && (
              <div className="p-3 rounded bg-white/[0.02] border border-white/5">
                <div className="font-bold text-foreground/60 mb-1">DISPERSION</div>
                {sentiment.dispersion_analysis}
              </div>
            )}
            {sentiment.asset_specific_notes && (
              <div className="p-3 rounded bg-white/[0.02] border border-white/5">
                <div className="font-bold text-foreground/60 mb-1">ASSET NOTES</div>
                {sentiment.asset_specific_notes}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

const AssetDetail = () => {
  const { symbol } = useParams();
  const navigate = useNavigate();
  const [price, setPrice] = useState(null);
  const [indicators, setIndicators] = useState(null);
  const [score, setScore] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [priceRes, indicatorsRes, scoreRes] = await Promise.allSettled([
          api.getLatestPrice(symbol),
          api.getIndicators(symbol),
          api.getScore(symbol)
        ]);

        if (priceRes.status === 'fulfilled') setPrice(priceRes.value.data);
        if (indicatorsRes.status === 'fulfilled') setIndicators(indicatorsRes.value.data);
        if (scoreRes.status === 'fulfilled') setScore(scoreRes.value.data);

        if (priceRes.status === 'rejected' && indicatorsRes.status === 'rejected' && scoreRes.status === 'rejected') {
          toast.error('No data available for this asset');
        }
      } catch (err) {
        console.error('Error fetching asset data:', err);
        toast.error('Failed to load asset data');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [symbol]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <Activity className="w-12 h-12 animate-pulse mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Loading asset data...</p>
        </div>
      </div>
    );
  }

  const scoreValue = score?.composite_score ?? 0;
  const breakdown = score?.breakdown || {};
  const hasData = !!(price || score || indicators);

  return (
    <div className="p-8" data-testid="asset-detail-page">
      <div className="mb-8">
        <button onClick={() => navigate('/')}
          className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition mb-4"
          data-testid="back-to-dashboard-btn">
          <ArrowLeft className="w-5 h-5" /> BACK TO DASHBOARD
        </button>
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-5xl font-bold tracking-tight mb-2" data-testid="asset-detail-symbol">{symbol}</h1>
            {price ? (
              <div className="flex items-baseline gap-3 flex-wrap mt-3">
                <span className="text-3xl font-data font-bold whitespace-nowrap" data-testid="asset-detail-price">{formatCurrency(price.price_usd, 'USD')}</span>
                <span className="text-xl font-data text-muted-foreground whitespace-nowrap">{formatCurrency(price.price_inr, 'INR')}</span>
              </div>
            ) : (
              <p className="text-muted-foreground mt-3">Price unavailable</p>
            )}
          </div>
          <div className={`px-6 py-3 rounded text-sm font-bold shrink-0 score-zone-${score?.zone || 'neutral'}`} data-testid="asset-detail-zone">
            {score?.zone ? getZoneLabel(score.zone) : 'NO DATA'}
          </div>
        </div>
      </div>

      {!hasData && (
        <div className="flex items-start gap-3 p-4 rounded glass-effect border border-destructive/30 mb-6">
          <AlertTriangle className="w-5 h-5 text-destructive shrink-0 mt-0.5" />
          <div>
            <p className="font-bold text-destructive mb-1">No market data available for {symbol}</p>
            <p className="text-sm text-muted-foreground">
              This symbol may not be recognised on Yahoo Finance. Try refreshing from the Dashboard,
              or check that the ticker is correct. Use the symbol as listed on Yahoo Finance.
            </p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="glass-effect rounded-sm p-6 col-span-1" data-testid="score-card">
          <div className="text-sm text-muted-foreground mb-2">DCA FAVORABILITY SCORE</div>
          <div className="text-6xl font-bold font-data mb-4" style={{ color: getScoreColor(scoreValue) }}>{scoreValue.toFixed(0)}</div>
          <div className="w-full bg-white/5 rounded-full h-3 mb-4 overflow-hidden">
            <div className="h-full transition-all" style={{ width: `${scoreValue}%`, backgroundColor: getScoreColor(scoreValue) }} />
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">{score?.explanation || 'Calculating explanation...'}</p>
        </div>

        <div className="glass-effect rounded-sm p-6 col-span-2" data-testid="score-breakdown">
          <div className="text-sm text-muted-foreground mb-4">SCORE BREAKDOWN</div>
          <div className="grid grid-cols-2 gap-4">
            {[
              { label: 'TECHNICAL & MOMENTUM', key: 'technical_momentum', color: 'bg-primary' },
              { label: 'VOLATILITY OPPORTUNITY', key: 'volatility_opportunity', color: 'bg-chart-3' },
              { label: 'STATISTICAL DEVIATION', key: 'statistical_deviation', color: 'bg-chart-4' },
              { label: 'MACRO & FX', key: 'macro_fx', color: 'bg-chart-2' },
            ].map(({ label, key, color }) => (
              <div key={key}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-muted-foreground">{label}</span>
                  <span className="font-data font-bold">{breakdown[key]?.toFixed(0) || 0}</span>
                </div>
                <div className="w-full bg-white/5 rounded-full h-2 overflow-hidden">
                  <div className={`h-full ${color} transition-all`} style={{ width: `${breakdown[key] || 0}%` }} />
                </div>
              </div>
            ))}
          </div>
          {score?.top_factors?.length > 0 && (
            <div className="mt-6 pt-6 border-t border-white/10">
              <div className="text-xs text-muted-foreground mb-3">KEY CONTRIBUTING FACTORS</div>
              <ul className="space-y-2 text-sm">
                {score.top_factors.map((factor, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <Zap className="w-4 h-4 text-primary flex-shrink-0 mt-0.5" />
                    <span>{factor}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      <SentimentPanel symbol={symbol} />

      <div className="glass-effect rounded-sm p-6 mb-6" data-testid="asset-news-section">
        <h2 className="text-xl font-bold mb-4">LATEST NEWS</h2>
        <AssetNewsWidget symbol={symbol} limit={3} compact={false} />
      </div>

      <div className="glass-effect rounded-sm p-6" data-testid="indicators-table">
        <h2 className="text-xl font-bold mb-6">TECHNICAL INDICATORS</h2>
        <IndicatorGrid indicators={indicators} />
      </div>
    </div>
  );
};

export default AssetDetail;
