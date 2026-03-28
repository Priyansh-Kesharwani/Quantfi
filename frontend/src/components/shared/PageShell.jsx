import React from 'react';
import { RefreshCw, TrendingUp } from 'lucide-react';

/**
 * PageShell — Standard page wrapper with header, loading, and empty states.
 *
 * Replaces 5× duplicated patterns for loading spinners, empty states,
 * and page headers across Dashboard, Assets, News, Settings, BacktestLab.
 *
 * @param {string} title — Page title
 * @param {string} [subtitle] — Subtitle text
 * @param {boolean} loading — Show loading spinner
 * @param {boolean} [isEmpty] — Show empty state
 * @param {string} [emptyTitle] — Empty state title
 * @param {string} [emptyMessage] — Empty state description
 * @param {Function} [onEmpty] — CTA action in empty state
 * @param {string} [emptyCTA] — CTA button label
 * @param {React.ReactNode} [actions] — Header action buttons
 * @param {React.ReactNode} children — Page content
 * @param {string} [testId] — data-testid for the main wrapper
 */
const PageShell = ({
  title,
  subtitle,
  loading,
  isEmpty,
  emptyTitle = 'NO ASSETS IN WATCHLIST',
  emptyMessage = 'Add assets to your watchlist to start analyzing DCA opportunities.',
  onEmpty,
  emptyCTA = 'ADD YOUR FIRST ASSET',
  actions,
  children,
  testId,
}) => {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen" data-testid={testId ? `loading-${testId}` : 'loading'}>
        <div className="text-center">
          <RefreshCw className="w-12 h-12 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Loading {title?.toLowerCase() || 'data'}...</p>
        </div>
      </div>
    );
  }

  if (isEmpty) {
    return (
      <div className="flex items-center justify-center h-screen" data-testid={testId ? `empty-${testId}` : 'empty'}>
        <div className="text-center max-w-md">
          <TrendingUp className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
          <h2 className="text-2xl font-bold mb-2">{emptyTitle}</h2>
          <p className="text-muted-foreground mb-6">{emptyMessage}</p>
          {onEmpty && (
            <button
              className="px-6 py-3 bg-primary text-primary-foreground rounded font-medium hover:bg-primary/90 transition"
              onClick={onEmpty}
              data-testid="add-first-asset-btn"
            >
              {emptyCTA}
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 lg:p-8 max-w-[1600px] mx-auto" data-testid={testId}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-4xl font-bold tracking-tight mb-1">{title}</h1>
          {subtitle && <p className="text-muted-foreground text-sm">{subtitle}</p>}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>
      {children}
    </div>
  );
};

export default PageShell;
