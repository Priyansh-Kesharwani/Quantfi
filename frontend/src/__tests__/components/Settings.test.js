import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import Settings from '../../pages/Settings';
import api from '../../api';

jest.mock('../../api');

const MOCK_SETTINGS = {
  data: {
    default_dca_cadence: 'monthly',
    default_dca_amount: 5000,
    dip_alert_threshold: 70,
    score_weights: {
      technical_momentum: 0.4,
      volatility_opportunity: 0.2,
      statistical_deviation: 0.2,
      macro_fx: 0.2
    }
  }
};

describe('Settings', () => {
  beforeEach(() => {
    api.getSettings.mockResolvedValue(MOCK_SETTINGS);
    api.updateSettings.mockResolvedValue({ data: {} });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('renders settings page after loading', async () => {
    render(<Settings />);
    await waitFor(() => {
      expect(screen.getByTestId('settings-page')).toBeInTheDocument();
    });
    expect(screen.getByTestId('settings-title')).toHaveTextContent('SETTINGS');
  });

  test('loads and displays DCA defaults', async () => {
    render(<Settings />);
    await waitFor(() => {
      expect(screen.getByTestId('default-cadence-select')).toHaveValue('monthly');
    });
    expect(screen.getByTestId('default-amount-input')).toHaveValue(5000);
    expect(screen.getByTestId('dip-threshold-input')).toHaveValue(70);
  });

  test('updates cadence selection', async () => {
    render(<Settings />);
    await waitFor(() => {
      expect(screen.getByTestId('default-cadence-select')).toBeInTheDocument();
    });
    fireEvent.change(screen.getByTestId('default-cadence-select'), { target: { value: 'weekly' } });
    expect(screen.getByTestId('default-cadence-select')).toHaveValue('weekly');
  });

  test('updates DCA amount input', async () => {
    render(<Settings />);
    await waitFor(() => {
      expect(screen.getByTestId('default-amount-input')).toBeInTheDocument();
    });
    fireEvent.change(screen.getByTestId('default-amount-input'), { target: { value: '10000' } });
    expect(screen.getByTestId('default-amount-input')).toHaveValue(10000);
  });

  test('save button triggers API call', async () => {
    render(<Settings />);
    await waitFor(() => {
      expect(screen.getByTestId('save-settings-btn')).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId('save-settings-btn'));
    await waitFor(() => {
      expect(api.updateSettings).toHaveBeenCalledTimes(1);
    });
  });

  test('displays total weight indicator', async () => {
    render(<Settings />);
    await waitFor(() => {
      expect(screen.getByText('TOTAL WEIGHT: 100%')).toBeInTheDocument();
    });
  });
});
