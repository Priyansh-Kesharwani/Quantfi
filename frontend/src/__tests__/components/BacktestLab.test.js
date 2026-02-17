import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import BacktestLab from '../../pages/BacktestLab';
import api from '../../api';

jest.mock('../../api');

const MOCK_ASSETS = {
  data: [
    { symbol: 'GOLD', name: 'Gold Futures' },
    { symbol: 'AAPL', name: 'Apple Inc.' },
  ]
};

const MOCK_BACKTEST_RESULT = {
  data: {
    total_invested: 60000,
    final_value_usd: 72000,
    total_return_pct: 20.0,
    annualized_return_pct: 10.5,
    num_regular_dca: 24,
    num_dip_buys: 3,
    total_units: 12.345,
    final_value_inr: 5990000,
    max_drawdown_pct: -12.5,
    avg_cost_basis: 1850.0,
    data_points: 8760,
    data_source: 'yfinance',
    data_start: '1991-01-02',
    data_end: '2026-02-07',
    equity_curve: [
      { date: '1991-06-01', portfolio_value: 5100, total_invested: 5000, price: 360, score: 55, is_dip_buy: false },
      { date: '2000-03-15', portfolio_value: 18000, total_invested: 15000, price: 800, score: 72, is_dip_buy: true },
      { date: '2026-01-01', portfolio_value: 72000, total_invested: 60000, price: 2050, score: 62, is_dip_buy: false },
    ]
  }
};

describe('BacktestLab', () => {
  beforeEach(() => {
    api.getAssets.mockResolvedValue(MOCK_ASSETS);
    api.runBacktest.mockResolvedValue(MOCK_BACKTEST_RESULT);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('renders backtest lab page', async () => {
    render(<BacktestLab />);
    await waitFor(() => {
      expect(screen.getByTestId('backtest-lab-page')).toBeInTheDocument();
    });
    expect(screen.getByTestId('backtest-title')).toHaveTextContent('BACKTEST LAB');
  });

  test('populates asset dropdown from API', async () => {
    render(<BacktestLab />);
    await waitFor(() => {
      expect(screen.getByTestId('backtest-asset-select')).toBeInTheDocument();
    });
    const options = screen.getByTestId('backtest-asset-select').querySelectorAll('option');
    expect(options.length).toBe(2);
  });

  test('shows empty results state initially', async () => {
    render(<BacktestLab />);
    await waitFor(() => {
      expect(screen.getByTestId('backtest-empty')).toBeInTheDocument();
    });
  });

  test('run backtest button triggers API call and shows results', async () => {
    render(<BacktestLab />);
    await waitFor(() => {
      expect(screen.getByTestId('run-backtest-btn')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId('run-backtest-btn'));
    await waitFor(() => {
      expect(api.runBacktest).toHaveBeenCalledTimes(1);
    });
    await waitFor(() => {
      expect(screen.getByTestId('backtest-results')).toBeInTheDocument();
    });
  });

  test('date inputs are interactive', async () => {
    render(<BacktestLab />);
    await waitFor(() => {
      expect(screen.getByTestId('backtest-start-date')).toBeInTheDocument();
    });
    fireEvent.change(screen.getByTestId('backtest-start-date'), { target: { value: '2023-01-01' } });
    expect(screen.getByTestId('backtest-start-date')).toHaveValue('2023-01-01');
  });
});
