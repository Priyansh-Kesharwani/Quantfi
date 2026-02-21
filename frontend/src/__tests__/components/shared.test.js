import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { ScoreBar, ZoneBadge, StatCard, PageShell, AssetPicker, FilterTabs, MetricGrid, RefreshButton } from '../../components/shared';
import { WatchlistProvider, useWatchlist } from '../../contexts/WatchlistContext';
import api from '../../api';
import { Activity } from 'lucide-react';

jest.mock('../../api');

/* ─── ScoreBar ─── */
describe('ScoreBar', () => {
  test('renders with value', () => {
    render(<ScoreBar value={72} />);
    expect(screen.getByTestId('score-bar')).toBeInTheDocument();
  });

  test('shows value when showValue=true', () => {
    render(<ScoreBar value={72} showValue />);
    expect(screen.getByText('72')).toBeInTheDocument();
  });

  test('shows label when provided', () => {
    render(<ScoreBar value={50} label="DCA SCORE" showValue />);
    expect(screen.getByText('DCA SCORE')).toBeInTheDocument();
  });

  test('clamps value to 0-100 range', () => {
    const { container } = render(<ScoreBar value={150} />);
    const bar = container.querySelector('[style]');
    expect(bar.style.width).toBe('100%');
  });

  test('handles zero value', () => {
    const { container } = render(<ScoreBar value={0} />);
    const bar = container.querySelector('[style]');
    expect(bar.style.width).toBe('0%');
  });
});

/* ─── ZoneBadge ─── */
describe('ZoneBadge', () => {
  test('renders strong_buy zone', () => {
    render(<ZoneBadge zone="strong_buy" />);
    expect(screen.getByText('STRONG BUY DIP')).toBeInTheDocument();
  });

  test('renders favorable zone', () => {
    render(<ZoneBadge zone="favorable" />);
    expect(screen.getByText('FAVORABLE')).toBeInTheDocument();
  });

  test('renders neutral zone', () => {
    render(<ZoneBadge zone="neutral" />);
    expect(screen.getByText('NEUTRAL')).toBeInTheDocument();
  });

  test('renders unfavorable zone', () => {
    render(<ZoneBadge zone="unfavorable" />);
    expect(screen.getByText('UNFAVORABLE')).toBeInTheDocument();
  });

  test('handles null zone gracefully', () => {
    render(<ZoneBadge zone={null} />);
    expect(screen.getByTestId('zone-badge')).toBeInTheDocument();
  });

  test('supports size variants', () => {
    const { rerender } = render(<ZoneBadge zone="neutral" size="sm" />);
    let badge = screen.getByTestId('zone-badge');
    expect(badge.className).toContain('text-[10px]');

    rerender(<ZoneBadge zone="neutral" size="lg" />);
    badge = screen.getByTestId('zone-badge');
    expect(badge.className).toContain('text-sm');
  });
});

/* ─── StatCard ─── */
describe('StatCard', () => {
  test('renders with label, value, and subValue', () => {
    render(<StatCard icon={Activity} label="TEST" value="42" subValue="details" />);
    expect(screen.getByText('TEST')).toBeInTheDocument();
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText('details')).toBeInTheDocument();
  });

  test('renders positive trend', () => {
    render(<StatCard icon={Activity} label="X" value="1" trend="Up" trendPositive={true} />);
    expect(screen.getByText('Up')).toBeInTheDocument();
    expect(screen.getByText('Up').className).toContain('text-emerald-400');
  });

  test('renders negative trend', () => {
    render(<StatCard icon={Activity} label="X" value="1" trend="Down" trendPositive={false} />);
    expect(screen.getByText('Down').className).toContain('text-red-400');
  });
});

/* ─── PageShell ─── */
describe('PageShell', () => {
  test('shows loading state', () => {
    render(<PageShell title="TEST" loading={true} testId="test-page" />);
    expect(screen.getByTestId('loading-test-page')).toBeInTheDocument();
  });

  test('shows empty state', () => {
    render(
      <PageShell
        title="TEST"
        loading={false}
        isEmpty={true}
        emptyTitle="NOTHING HERE"
        onEmpty={jest.fn()}
        testId="test-page"
      />
    );
    expect(screen.getByTestId('empty-test-page')).toBeInTheDocument();
    expect(screen.getByText('NOTHING HERE')).toBeInTheDocument();
  });

  test('renders children when loaded and not empty', () => {
    render(
      <PageShell title="MY PAGE" subtitle="Sub" loading={false} isEmpty={false} testId="test-page">
        <div data-testid="child">Content</div>
      </PageShell>
    );
    expect(screen.getByTestId('test-page')).toBeInTheDocument();
    expect(screen.getByText('MY PAGE')).toBeInTheDocument();
    expect(screen.getByText('Sub')).toBeInTheDocument();
    expect(screen.getByTestId('child')).toBeInTheDocument();
  });

  test('renders action buttons', () => {
    render(
      <PageShell title="P" loading={false} isEmpty={false} actions={<button data-testid="act">Go</button>}>
        <div>c</div>
      </PageShell>
    );
    expect(screen.getByTestId('act')).toBeInTheDocument();
  });
});

/* ─── AssetPicker ─── */
describe('AssetPicker', () => {
  const assets = [
    { symbol: 'AAPL', name: 'Apple Inc.', asset_type: 'equity' },
    { symbol: 'GOOG', name: 'Alphabet', asset_type: 'equity' },
  ];

  test('renders options for each asset', () => {
    render(<AssetPicker assets={assets} value="AAPL" onChange={jest.fn()} />);
    expect(screen.getByTestId('asset-picker')).toBeInTheDocument();
    expect(screen.getByText(/AAPL/)).toBeInTheDocument();
    expect(screen.getByText(/GOOG/)).toBeInTheDocument();
  });

  test('calls onChange when selection changes', () => {
    const onChange = jest.fn();
    render(<AssetPicker assets={assets} value="AAPL" onChange={onChange} />);
    fireEvent.change(screen.getByTestId('asset-picker'), { target: { value: 'GOOG' } });
    expect(onChange).toHaveBeenCalledWith('GOOG');
  });

  test('shows label when provided', () => {
    render(<AssetPicker assets={assets} value="AAPL" onChange={jest.fn()} label="ASSET" />);
    expect(screen.getByText('ASSET')).toBeInTheDocument();
  });

  test('handles empty asset list', () => {
    render(<AssetPicker assets={[]} value="" onChange={jest.fn()} />);
    expect(screen.getByText(/No assets/)).toBeInTheDocument();
  });
});

/* ─── FilterTabs ─── */
describe('FilterTabs', () => {
  const options = [
    { key: 'all', label: 'ALL' },
    { key: 'equity', label: 'EQUITY' },
    { key: 'crypto', label: 'CRYPTO' },
  ];

  test('renders all options', () => {
    render(<FilterTabs options={options} value="all" onChange={jest.fn()} />);
    expect(screen.getByText('ALL')).toBeInTheDocument();
    expect(screen.getByText('EQUITY')).toBeInTheDocument();
    expect(screen.getByText('CRYPTO')).toBeInTheDocument();
  });

  test('highlights active option', () => {
    render(<FilterTabs options={options} value="equity" onChange={jest.fn()} />);
    const btn = screen.getByTestId('filter-tab-equity');
    expect(btn.className).toContain('bg-primary');
  });

  test('fires onChange on click', () => {
    const onChange = jest.fn();
    render(<FilterTabs options={options} value="all" onChange={onChange} />);
    fireEvent.click(screen.getByTestId('filter-tab-crypto'));
    expect(onChange).toHaveBeenCalledWith('crypto');
  });
});

/* ─── MetricGrid ─── */
describe('MetricGrid', () => {
  test('renders items', () => {
    const items = [
      { label: 'RSI', value: '45.2' },
      { label: 'ATR', value: '35.0' },
    ];
    render(<MetricGrid items={items} />);
    expect(screen.getByText('RSI')).toBeInTheDocument();
    expect(screen.getByText('45.2')).toBeInTheDocument();
    expect(screen.getByText('ATR')).toBeInTheDocument();
  });

  test('skips null values', () => {
    const items = [
      { label: 'RSI', value: '45' },
      { label: 'EMPTY', value: null },
    ];
    render(<MetricGrid items={items} />);
    expect(screen.getByText('RSI')).toBeInTheDocument();
    expect(screen.queryByText('EMPTY')).not.toBeInTheDocument();
  });
});

/* ─── RefreshButton ─── */
describe('RefreshButton', () => {
  test('renders with default label', () => {
    render(<RefreshButton onClick={jest.fn()} />);
    expect(screen.getByText('REFRESH')).toBeInTheDocument();
  });

  test('fires onClick', () => {
    const onClick = jest.fn();
    render(<RefreshButton onClick={onClick} />);
    fireEvent.click(screen.getByTestId('refresh-btn'));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  test('shows spinner when loading', () => {
    const { container } = render(<RefreshButton onClick={jest.fn()} loading={true} />);
    expect(container.querySelector('.animate-spin')).toBeInTheDocument();
  });

  test('is disabled when loading', () => {
    render(<RefreshButton onClick={jest.fn()} loading={true} />);
    expect(screen.getByTestId('refresh-btn')).toBeDisabled();
  });
});

/* ─── WatchlistContext ─── */
describe('WatchlistContext', () => {
  const MOCK_DATA = {
    data: {
      assets: [
        {
          asset: { symbol: 'AAPL', name: 'Apple', asset_type: 'equity' },
          price: { price_usd: 180, price_inr: 15000 },
          score: { composite_score: 72, zone: 'favorable' },
          indicators: { rsi_14: 45 },
        },
      ],
    },
  };

  beforeEach(() => {
    api.getDashboard.mockResolvedValue(MOCK_DATA);
  });
  afterEach(() => jest.clearAllMocks());

  function TestConsumer() {
    const { assets, assetList, loading } = useWatchlist();
    if (loading) return <div data-testid="ctx-loading">Loading</div>;
    return (
      <div data-testid="ctx-loaded">
        <span data-testid="ctx-count">{assets.length}</span>
        <span data-testid="ctx-first">{assetList[0]?.symbol}</span>
      </div>
    );
  }

  test('provides assets after loading', async () => {
    render(
      <BrowserRouter>
        <WatchlistProvider>
          <TestConsumer />
        </WatchlistProvider>
      </BrowserRouter>
    );
    expect(screen.getByTestId('ctx-loading')).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByTestId('ctx-loaded')).toBeInTheDocument();
    });
    expect(screen.getByTestId('ctx-count')).toHaveTextContent('1');
    expect(screen.getByTestId('ctx-first')).toHaveTextContent('AAPL');
  });

  test('calls getDashboard once on mount', async () => {
    render(
      <BrowserRouter>
        <WatchlistProvider>
          <TestConsumer />
        </WatchlistProvider>
      </BrowserRouter>
    );
    await waitFor(() => {
      expect(api.getDashboard).toHaveBeenCalledTimes(1);
    });
  });
});
