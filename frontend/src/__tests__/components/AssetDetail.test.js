import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import AssetDetail from '../../pages/AssetDetail';
import api from '../../api';

jest.mock('../../api');

const MOCK_PRICE = {
  data: { price_usd: 2650.50, price_inr: 220000, timestamp: new Date().toISOString() }
};
const MOCK_INDICATORS = {
  data: {
    sma_50: 2600, sma_200: 2400, rsi_14: 45.2, macd: 12.5,
    atr_14: 35.0, adx_14: 28.0, drawdown_pct: -3.5, z_score_50: -0.8,
    bb_upper: 2700, bb_lower: 2500
  }
};
const MOCK_SCORE = {
  data: {
    composite_score: 72,
    zone: 'favorable',
    explanation: 'RSI indicates oversold conditions with strong momentum reversal signals.',
    breakdown: { technical_momentum: 80, volatility_opportunity: 65, statistical_deviation: 70, macro_fx: 60 },
    top_factors: ['RSI below 30 — oversold', 'Price below SMA-200']
  }
};

const renderAssetDetail = (symbol = 'GOLD') => {
  return render(
    <MemoryRouter initialEntries={[`/assets/${symbol}`]}>
      <Routes>
        <Route path="/assets/:symbol" element={<AssetDetail />} />
      </Routes>
    </MemoryRouter>
  );
};

describe('AssetDetail', () => {
  beforeEach(() => {
    api.getLatestPrice.mockResolvedValue(MOCK_PRICE);
    api.getIndicators.mockResolvedValue(MOCK_INDICATORS);
    api.getScore.mockResolvedValue(MOCK_SCORE);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('renders asset detail page with symbol', async () => {
    renderAssetDetail('GOLD');
    await waitFor(() => {
      expect(screen.getByTestId('asset-detail-page')).toBeInTheDocument();
    });
    expect(screen.getByTestId('asset-detail-symbol')).toHaveTextContent('GOLD');
  });

  test('displays price data', async () => {
    renderAssetDetail('GOLD');
    await waitFor(() => {
      expect(screen.getByTestId('asset-detail-price')).toBeInTheDocument();
    });
  });

  test('displays score breakdown bars', async () => {
    renderAssetDetail('GOLD');
    await waitFor(() => {
      expect(screen.getByTestId('score-breakdown')).toBeInTheDocument();
    });
  });

  test('displays score explanation', async () => {
    renderAssetDetail('GOLD');
    await waitFor(() => {
      expect(screen.getByText(/RSI indicates oversold/)).toBeInTheDocument();
    });
  });

  test('displays top contributing factors', async () => {
    renderAssetDetail('GOLD');
    await waitFor(() => {
      expect(screen.getByText('RSI below 30 — oversold')).toBeInTheDocument();
    });
  });

  test('renders technical indicators grid', async () => {
    renderAssetDetail('GOLD');
    await waitFor(() => {
      expect(screen.getByTestId('indicators-table')).toBeInTheDocument();
    });
  });

  test('renders news section', async () => {
    renderAssetDetail('GOLD');
    await waitFor(() => {
      expect(screen.getByTestId('asset-news-section')).toBeInTheDocument();
    });
  });

  test('back button is present', async () => {
    renderAssetDetail('GOLD');
    await waitFor(() => {
      expect(screen.getByTestId('back-to-dashboard-btn')).toBeInTheDocument();
    });
  });
});
