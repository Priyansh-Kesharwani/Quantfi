import React from 'react';
import { RefreshCw } from 'lucide-react';

/**
 * RefreshButton — Consistent refresh trigger.
 *
 * Replaces 4× identical refresh buttons across
 * Dashboard, Assets, News, and AssetDetail.
 *
 * @param {Function} onClick — Refresh handler
 * @param {boolean} [loading=false] — Show spinner
 * @param {string} [label='REFRESH'] — Button label
 * @param {string} [className] — Additional classes
 */
const RefreshButton = ({ onClick, loading = false, label = 'REFRESH', className = '' }) => (
  <button
    onClick={onClick}
    disabled={loading}
    className={`flex items-center gap-2 px-4 py-2 glass-effect hover:bg-white/10 rounded transition ${className}`}
    data-testid="refresh-btn"
  >
    <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
    <span className="text-sm">{label}</span>
  </button>
);

export default RefreshButton;
