import React from 'react';

import AssetNewsWidget from '@/components/AssetNewsWidget';

import { getScoreColor, getZoneLabel } from '@/utils';

const AssetCard = ({ asset, score, price, indicators, onClick }) => {
  if (!asset) return null;

  const scoreValue = score?.composite_score || 0;
  const zone = score?.zone || 'neutral';
  const priceUSD = price?.price_usd || 0;
  const priceINR = price?.price_inr || 0;
  const rsi = indicators?.rsi_14;
  const drawdown = indicators?.drawdown_pct;

  return (
    <div
      className="glass-effect rounded-sm p-6 hover:border-primary/30 transition-all h-full cursor-pointer"
      onClick={onClick}
      data-testid={`asset-card-${asset.symbol}`}
    >
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
          <div className="h-full transition-all duration-300"
            style={{ width: `${scoreValue}%`, backgroundColor: getScoreColor(scoreValue) }} />
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

      <AssetNewsWidget symbol={asset.symbol} compact={true} limit={1} />
    </div>
  );
};

export default AssetCard;
