import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import api from '../api';
import { formatCurrency, formatNumber, formatPercent, getScoreColor, getZoneLabel } from '../utils';
import { ArrowLeft, Activity, Zap } from 'lucide-react';
import { toast } from 'sonner';

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
        const [priceRes, indicatorsRes, scoreRes] = await Promise.all([
          api.getLatestPrice(symbol),
          api.getIndicators(symbol),
          api.getScore(symbol)
        ]);

        setPrice(priceRes.data);
        setIndicators(indicatorsRes.data);
        setScore(scoreRes.data);
      } catch (error) {
        console.error('Error fetching asset data:', error);
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

  const scoreValue = score?.composite_score || 0;
  const breakdown = score?.breakdown || {};

  return (
    <div className="p-8" data-testid="asset-detail-page">
      {/* Header */}
      <div className="mb-8">
        <button
          onClick={() => navigate('/')}
          className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition mb-4"
          data-testid="back-to-dashboard-btn"
        >
          <ArrowLeft className="w-5 h-5" />
          BACK TO DASHBOARD
        </button>
        
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-5xl font-bold tracking-tight mb-2" data-testid="asset-detail-symbol">{symbol}</h1>
            <div className="flex items-center gap-4 mt-3">
              <span className="text-3xl font-data font-bold" data-testid="asset-detail-price">
                {formatCurrency(price?.price_usd, 'USD')}
              </span>
              <span className="text-xl font-data text-muted-foreground">
                {formatCurrency(price?.price_inr, 'INR')}
              </span>
            </div>
          </div>
          
          <div className={`px-6 py-3 rounded text-sm font-bold score-zone-${score?.zone}`} data-testid="asset-detail-zone">
            {getZoneLabel(score?.zone)}
          </div>
        </div>
      </div>

      {/* DCA Score Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Main Score */}
        <div className="glass-effect rounded-sm p-6 col-span-1" data-testid="score-card">
          <div className="text-sm text-muted-foreground mb-2">DCA FAVORABILITY SCORE</div>
          <div className="text-6xl font-bold font-data mb-4" style={{ color: getScoreColor(scoreValue) }}>
            {scoreValue.toFixed(0)}
          </div>
          <div className="w-full bg-white/5 rounded-full h-3 mb-4 overflow-hidden">
            <div
              className="h-full transition-all"
              style={{ width: `${scoreValue}%`, backgroundColor: getScoreColor(scoreValue) }}
            />
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {score?.explanation || 'Calculating explanation...'}
          </p>
        </div>

        {/* Score Breakdown */}
        <div className="glass-effect rounded-sm p-6 col-span-2" data-testid="score-breakdown">
          <div className="text-sm text-muted-foreground mb-4">SCORE BREAKDOWN</div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-muted-foreground">TECHNICAL & MOMENTUM</span>
                <span className="font-data font-bold">{breakdown.technical_momentum?.toFixed(0) || 0}</span>
              </div>
              <div className="w-full bg-white/5 rounded-full h-2 overflow-hidden">
                <div
                  className="h-full bg-primary transition-all"
                  style={{ width: `${breakdown.technical_momentum || 0}%` }}
                />
              </div>
            </div>
            
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-muted-foreground">VOLATILITY OPPORTUNITY</span>
                <span className="font-data font-bold">{breakdown.volatility_opportunity?.toFixed(0) || 0}</span>
              </div>
              <div className="w-full bg-white/5 rounded-full h-2 overflow-hidden">
                <div
                  className="h-full bg-chart-3 transition-all"
                  style={{ width: `${breakdown.volatility_opportunity || 0}%` }}
                />
              </div>
            </div>
            
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-muted-foreground">STATISTICAL DEVIATION</span>
                <span className="font-data font-bold">{breakdown.statistical_deviation?.toFixed(0) || 0}</span>
              </div>
              <div className="w-full bg-white/5 rounded-full h-2 overflow-hidden">
                <div
                  className="h-full bg-chart-4 transition-all"
                  style={{ width: `${breakdown.statistical_deviation || 0}%` }}
                />
              </div>
            </div>
            
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-muted-foreground">MACRO & FX</span>
                <span className="font-data font-bold">{breakdown.macro_fx?.toFixed(0) || 0}</span>
              </div>
              <div className="w-full bg-white/5 rounded-full h-2 overflow-hidden">
                <div
                  className="h-full bg-chart-2 transition-all"
                  style={{ width: `${breakdown.macro_fx || 0}%` }}
                />
              </div>
            </div>
          </div>
          
          {score?.top_factors && score.top_factors.length > 0 && (
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

      {/* Technical Indicators */}
      <div className="glass-effect rounded-sm p-6" data-testid="indicators-table">
        <h2 className="text-xl font-bold mb-6">TECHNICAL INDICATORS</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-6">
          {indicators?.sma_50 && (
            <div>
              <div className="text-xs text-muted-foreground mb-1">SMA 50</div>
              <div className="font-data font-semibold">{formatCurrency(indicators.sma_50, 'USD')}</div>
            </div>
          )}
          {indicators?.sma_200 && (
            <div>
              <div className="text-xs text-muted-foreground mb-1">SMA 200</div>
              <div className="font-data font-semibold">{formatCurrency(indicators.sma_200, 'USD')}</div>
            </div>
          )}
          {indicators?.rsi_14 !== null && indicators?.rsi_14 !== undefined && (
            <div>
              <div className="text-xs text-muted-foreground mb-1">RSI (14)</div>
              <div className="font-data font-semibold">{formatNumber(indicators.rsi_14, 1)}</div>
            </div>
          )}
          {indicators?.macd !== null && indicators?.macd !== undefined && (
            <div>
              <div className="text-xs text-muted-foreground mb-1">MACD</div>
              <div className="font-data font-semibold">{formatNumber(indicators.macd, 2)}</div>
            </div>
          )}
          {indicators?.atr_14 && (
            <div>
              <div className="text-xs text-muted-foreground mb-1">ATR (14)</div>
              <div className="font-data font-semibold">{formatNumber(indicators.atr_14, 2)}</div>
            </div>
          )}
          {indicators?.adx_14 && (
            <div>
              <div className="text-xs text-muted-foreground mb-1">ADX (14)</div>
              <div className="font-data font-semibold">{formatNumber(indicators.adx_14, 1)}</div>
            </div>
          )}
          {indicators?.drawdown_pct !== null && indicators?.drawdown_pct !== undefined && (
            <div>
              <div className="text-xs text-muted-foreground mb-1">DRAWDOWN</div>
              <div className="font-data font-semibold text-destructive">{formatPercent(indicators.drawdown_pct, 1)}</div>
            </div>
          )}
          {indicators?.z_score_50 !== null && indicators?.z_score_50 !== undefined && (
            <div>
              <div className="text-xs text-muted-foreground mb-1">Z-SCORE (50)</div>
              <div className="font-data font-semibold">{formatNumber(indicators.z_score_50, 2)}</div>
            </div>
          )}
          {indicators?.bb_upper && (
            <div>
              <div className="text-xs text-muted-foreground mb-1">BB UPPER</div>
              <div className="font-data font-semibold">{formatCurrency(indicators.bb_upper, 'USD')}</div>
            </div>
          )}
          {indicators?.bb_lower && (
            <div>
              <div className="text-xs text-muted-foreground mb-1">BB LOWER</div>
              <div className="font-data font-semibold">{formatCurrency(indicators.bb_lower, 'USD')}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AssetDetail;
