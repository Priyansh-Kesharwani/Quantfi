import React, { createContext, useContext, useState, useEffect, useCallback, useMemo } from 'react';
import api from '../api';

const WatchlistContext = createContext(null);

/**
 * WatchlistProvider — Single source of truth for watchlist data.
 *
 * Replaces the duplicated `api.getDashboard()` / `api.getAssets()` calls
 * across Dashboard, Assets, BacktestLab, and News pages.
 * Also eliminates the `refreshKey` prop-drilling anti-pattern.
 */
export function WatchlistProvider({ children }) {
  const [dashboardData, setDashboardData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    setError(null);
    try {
      const response = await api.getDashboard();
      setDashboardData(response.data.assets || []);
    } catch (err) {
      console.error('[WatchlistContext] Error fetching dashboard:', err);
      setError(err?.response?.data?.detail || err?.message || 'Failed to load watchlist');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Listen for global addAsset / refreshDashboard events
  useEffect(() => {
    const handleRefresh = () => fetchData(true);
    window.addEventListener('refreshDashboard', handleRefresh);
    return () => window.removeEventListener('refreshDashboard', handleRefresh);
  }, [fetchData]);

  const refresh = useCallback(() => {
    fetchData(true);
  }, [fetchData]);

  const removeAsset = useCallback(async (symbol) => {
    await api.removeAsset(symbol);
    setDashboardData(prev => prev.filter(item => item.asset.symbol !== symbol));
  }, []);

  // Derived: simple asset list for dropdowns / selectors
  const assetList = useMemo(() =>
    dashboardData.map(item => item.asset),
    [dashboardData]
  );

  const value = useMemo(() => ({
    assets: dashboardData,
    assetList,
    loading,
    refreshing,
    error,
    refresh,
    removeAsset,
  }), [dashboardData, assetList, loading, refreshing, error, refresh, removeAsset]);

  return (
    <WatchlistContext.Provider value={value}>
      {children}
    </WatchlistContext.Provider>
  );
}

/**
 * useWatchlist — Hook to access watchlist data from any component.
 */
export function useWatchlist() {
  const ctx = useContext(WatchlistContext);
  if (!ctx) {
    throw new Error('useWatchlist must be used within a WatchlistProvider');
  }
  return ctx;
}

export default WatchlistContext;
