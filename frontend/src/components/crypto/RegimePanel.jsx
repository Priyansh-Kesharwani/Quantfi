import React from 'react';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

const REGIME_COLORS = {
  TRENDING: '#3b82f6',
  RANGING: '#eab308',
  STRESS: '#ef4444',
};

const REGIME_TEXT = {
  TRENDING: 'text-blue-400',
  RANGING: 'text-yellow-400',
  STRESS: 'text-red-400',
};

export default function RegimePanel({ regimeCounts }) {
  if (!regimeCounts) return null;

  const total = Object.values(regimeCounts).reduce((a, b) => a + b, 0);
  if (total === 0) return null;

  return (
    <Card className="glass-effect">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">REGIMES</CardTitle>
          <span className="text-xs px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 font-mono">
            HMM
          </span>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <div className="flex h-3 rounded-full overflow-hidden gap-0.5">
          {Object.entries(regimeCounts).map(([regime, count]) => (
            <div
              key={regime}
              style={{
                width: `${((count / total) * 100).toFixed(1)}%`,
                backgroundColor: REGIME_COLORS[regime] || '#666',
              }}
              className="rounded-sm"
              title={`${regime}: ${count}`}
            />
          ))}
        </div>
        {Object.entries(regimeCounts).map(([regime, count]) => (
          <div key={regime} className="flex justify-between">
            <span className={`font-medium ${REGIME_TEXT[regime] || 'text-muted-foreground'}`}>
              {regime}
            </span>
            <span>
              {count} bars ({((count / total) * 100).toFixed(0)}%)
            </span>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
