import React from 'react';
import { getZoneLabel } from '../../utils';

/**
 * ZoneBadge — Colored zone label badge.
 *
 * Replaces 5× identical zone badge implementations across
 * AssetCard, Assets (grid+table), AssetDetail, and Dashboard QuickGlance.
 *
 * @param {string} zone — Zone key ('strong_buy', 'favorable', 'neutral', 'unfavorable')
 * @param {string} [size='sm'] — 'sm' for compact, 'md' for default, 'lg' for detail page
 */
const ZoneBadge = ({ zone, size = 'sm' }) => {
  const sizeClasses = {
    sm: 'text-[10px] px-2 py-0.5',
    md: 'text-xs px-2.5 py-1',
    lg: 'text-sm px-6 py-3',
  };

  return (
    <span
      className={`${sizeClasses[size] || sizeClasses.sm} rounded font-bold whitespace-nowrap score-zone-${zone || 'neutral'}`}
      data-testid="zone-badge"
    >
      {getZoneLabel(zone)}
    </span>
  );
};

export default ZoneBadge;
