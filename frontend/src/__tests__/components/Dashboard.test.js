import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import Dashboard from '../../pages/Dashboard';
import api from '../../api';

// Mock the api module
jest.mock('../../api');

const MOCK_DASHBOARD = {
  data: {
    assets: [
      {
        asset: { symbol: 'GOLD', name: 'Gold Futures', asset_type: 'commodity' },
        score: { composite_score: 72, zone: 'favorable' },
        price: { price_usd: 2650.50, price_inr: 220000.00 },
        indicators: { rsi_14: 45.2, drawdown_pct: -3.5 }
      },
      {
        asset: { symbol: 'AAPL', name: 'Apple Inc.', asset_type: 'equity' },
        score: { composite_score: 55, zone: 'neutral' },
        price: { price_usd: 185.30, price_inr: 15400.00 },
        indicators: { rsi_14: 60.1, drawdown_pct: -8.2 }
      }
    ]
  }
};

const MOCK_NEWS = {
  data: { news: [] }
};

const renderDashboard = (props = {}) => {
  return render(
    <BrowserRouter>
      <Dashboard refreshKey={0} {...props} />
    </BrowserRouter>
  );
};

describe('Dashboard', () => {
  beforeEach(() => {
    api.getDashboard.mockResolvedValue(MOCK_DASHBOARD);
    api.getNews.mockResolvedValue(MOCK_NEWS);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('shows loading state initially', () => {
    api.getDashboard.mockReturnValue(new Promise(() => {})); // never resolves
    renderDashboard();
    expect(screen.getByTestId('loading-dashboard')).toBeInTheDocument();
  });

  test('renders dashboard main after loading', async () => {
    renderDashboard();
    await waitFor(() => {
      expect(screen.getByTestId('dashboard-main')).toBeInTheDocument();
    });
  });

  test('displays dashboard title', async () => {
    renderDashboard();
    await waitFor(() => {
      expect(screen.getByTestId('dashboard-title')).toHaveTextContent('DASHBOARD');
    });
  });

  test('refresh button re-fetches dashboard data', async () => {
    renderDashboard();
    await waitFor(() => {
      expect(screen.getByTestId('dashboard-main')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId('refresh-dashboard-btn'));
    expect(api.getDashboard).toHaveBeenCalledTimes(2); // initial + refresh
  });

  test('shows empty state when no assets', async () => {
    api.getDashboard.mockResolvedValue({ data: { assets: [] } });
    renderDashboard();
    await waitFor(() => {
      expect(screen.getByTestId('empty-dashboard')).toBeInTheDocument();
    });
    expect(screen.getByTestId('add-first-asset-btn')).toBeInTheDocument();
  });

  test('renders info banner about DCA scoring', async () => {
    renderDashboard();
    await waitFor(() => {
      expect(screen.getByTestId('info-banner')).toBeInTheDocument();
    });
  });

  test('re-fetches when refreshKey changes', async () => {
    const { rerender } = renderDashboard({ refreshKey: 0 });
    await waitFor(() => {
      expect(api.getDashboard).toHaveBeenCalledTimes(1);
    });

    rerender(
      <BrowserRouter>
        <Dashboard refreshKey={1} />
      </BrowserRouter>
    );
    await waitFor(() => {
      expect(api.getDashboard).toHaveBeenCalledTimes(2);
    });
  });
});
