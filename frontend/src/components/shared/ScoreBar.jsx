import React from 'react';
import { getScoreColor } from '../../utils';

/**
 * ScoreBar — Colored progress bar for DCA/Entry/Exit scores.
 *
 * Replaces 4× identical hand-rolled progress bars across
 * AssetCard, Assets (grid+table), AssetDetail, and Dashboard.
 *
 * @param {number} value — Score value 0-100
 * @param {string} [height='h-1.5'] — Tailwind height class
 * @param {boolean} [showValue=false] — Show numeric value beside bar
 * @param {string} [label] — Optional label (e.g. 'DCA SCORE')
 * @param {string} [className] — Additional container classes
 */
const ScoreBar = ({ value = 0, height = 'h-1.5', showValue = false, label, className = '' }) => {
  const color = getScoreColor(value);
  const safeVal = Math.max(0, Math.min(100, value));

  return (
    <div className={className} data-testid="score-bar">
      {(label || showValue) && (
        <div className="flex items-center justify-between mb-1.5">
          {label && <span className="text-xs text-muted-foreground">{label}</span>}
          {showValue && (
            <span className="font-data font-bold text-sm" style={{ color }}>
              {safeVal.toFixed(0)}
            </span>
          )}
        </div>
      )}
      <div className={`w-full bg-white/5 rounded-full ${height} overflow-hidden`}>
        <div
          className="h-full transition-all duration-300 rounded-full"
          style={{ width: `${safeVal}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
};

export default ScoreBar;
