import React from 'react';

/**
 * MetricGrid — Generic grid of label/value pairs.
 *
 * Replaces IndicatorGrid in AssetDetail and inline metric grids in
 * BacktestLab, and will be used in Signals, ValidationLab, etc.
 *
 * @param {Array} items — [{label, value, format?, color?, suffix?}]
 * @param {number} [cols=4] — Number of columns (responsive)
 * @param {string} [className] — Additional classes
 */
const MetricGrid = ({ items = [], cols = 4, className = '' }) => {
  const colsClass = {
    2: 'grid-cols-2',
    3: 'grid-cols-2 md:grid-cols-3',
    4: 'grid-cols-2 md:grid-cols-4',
    6: 'grid-cols-2 md:grid-cols-4 lg:grid-cols-6',
  };

  return (
    <div className={`grid ${colsClass[cols] || colsClass[4]} gap-6 ${className}`} data-testid="metric-grid">
      {items.filter(Boolean).map((item, idx) => {
        if (item.value === null || item.value === undefined) return null;
        return (
          <div key={item.label || idx}>
            <div className="text-xs text-muted-foreground mb-1">{item.label}</div>
            <div
              className="font-data font-semibold"
              style={item.color ? { color: item.color } : undefined}
            >
              {item.value}{item.suffix || ''}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default MetricGrid;
