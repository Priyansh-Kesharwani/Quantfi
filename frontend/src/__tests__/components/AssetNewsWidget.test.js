import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import AssetNewsWidget from '../../components/AssetNewsWidget';

// Mock the entire newsService module
jest.mock('../../services/newsService', () => ({
  getNewsForAsset: jest.fn(),
  getSentimentColor: jest.fn(() => 'text-zinc-400 bg-zinc-400/10'),
  formatRelativeTime: jest.fn(() => '2h ago'),
}));

// Import mocked functions for control
const { getNewsForAsset, getSentimentColor, formatRelativeTime } = require('../../services/newsService');

describe('AssetNewsWidget', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  test('shows loading skeleton initially', () => {
    getNewsForAsset.mockReturnValue(new Promise(() => {})); // never resolves
    render(<AssetNewsWidget symbol="GOLD" />);
    expect(screen.getByTestId('news-widget-loading-GOLD')).toBeInTheDocument();
  });

  test('renders headlines when data is available', async () => {
    getNewsForAsset.mockResolvedValue([
      { title: 'Gold surges 2% on safe-haven demand', source: 'Reuters', timestamp: new Date().toISOString(), sentiment: 'positive' },
      { title: 'Central banks buy record gold', source: 'Bloomberg', timestamp: new Date().toISOString(), sentiment: 'positive' },
    ]);
    getSentimentColor.mockReturnValue('text-emerald-400 bg-emerald-400/10');
    formatRelativeTime.mockReturnValue('2h ago');

    render(<AssetNewsWidget symbol="GOLD" limit={2} />);

    await waitFor(() => {
      expect(screen.getByTestId('news-widget-GOLD')).toBeInTheDocument();
    });
    expect(screen.getByTestId('news-headline-GOLD-0')).toBeInTheDocument();
    expect(screen.getByTestId('news-headline-GOLD-1')).toBeInTheDocument();
    expect(screen.getByText('Gold surges 2% on safe-haven demand')).toBeInTheDocument();
  });

  test('compact mode shows only 1 headline', async () => {
    getNewsForAsset.mockResolvedValue([
      { title: 'Headline 1', source: 'Src1', timestamp: new Date().toISOString(), sentiment: 'neutral' },
      { title: 'Headline 2', source: 'Src2', timestamp: new Date().toISOString(), sentiment: 'neutral' },
    ]);

    render(<AssetNewsWidget symbol="AAPL" compact={true} limit={2} />);

    await waitFor(() => {
      expect(screen.getByTestId('news-widget-AAPL')).toBeInTheDocument();
    });
    expect(screen.getByTestId('news-headline-AAPL-0')).toBeInTheDocument();
    expect(screen.queryByTestId('news-headline-AAPL-1')).not.toBeInTheDocument();
  });

  test('shows empty state when no headlines returned', async () => {
    getNewsForAsset.mockResolvedValue([]);

    render(<AssetNewsWidget symbol="UNKNOWN" />);

    await waitFor(() => {
      expect(screen.getByTestId('news-widget-empty-UNKNOWN')).toBeInTheDocument();
    });
    expect(screen.getByText('No news available')).toBeInTheDocument();
  });

  test('handles API error gracefully and shows empty state', async () => {
    getNewsForAsset.mockRejectedValue(new Error('Network error'));

    render(<AssetNewsWidget symbol="ERR" />);

    await waitFor(() => {
      expect(screen.getByTestId('news-widget-empty-ERR')).toBeInTheDocument();
    });
    expect(screen.getByText('No news available')).toBeInTheDocument();
  });
});
