import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import Sidebar from '../../components/Sidebar';

const renderSidebar = (props = {}) => {
  const defaultProps = { onAddAsset: jest.fn() };
  return render(
    <BrowserRouter>
      <Sidebar {...defaultProps} {...props} />
    </BrowserRouter>
  );
};

describe('Sidebar', () => {
  test('renders the QuantFi app title', () => {
    renderSidebar();
    expect(screen.getByTestId('app-title')).toHaveTextContent('QUANTFI');
  });

  test('renders all 6 navigation items (including Validation Lab)', () => {
    renderSidebar();
    expect(screen.getByTestId('nav-dashboard')).toBeInTheDocument();
    expect(screen.getByTestId('nav-assets')).toBeInTheDocument();
    expect(screen.getByTestId('nav-backtest-lab')).toBeInTheDocument();
    expect(screen.getByTestId('nav-validation-lab')).toBeInTheDocument();
    expect(screen.getByTestId('nav-news-&-events')).toBeInTheDocument();
    expect(screen.getByTestId('nav-settings')).toBeInTheDocument();
  });

  test('each nav item has a label', () => {
    renderSidebar();
    expect(screen.getByText('DASHBOARD')).toBeInTheDocument();
    expect(screen.getByText('ASSETS')).toBeInTheDocument();
    expect(screen.getByText('BACKTEST LAB')).toBeInTheDocument();
    expect(screen.getByText('VALIDATION LAB')).toBeInTheDocument();
    expect(screen.getByText('NEWS & EVENTS')).toBeInTheDocument();
    expect(screen.getByText('SETTINGS')).toBeInTheDocument();
  });

  test('renders the Add Asset button', () => {
    renderSidebar();
    const btn = screen.getByTestId('add-asset-btn');
    expect(btn).toBeInTheDocument();
    expect(btn).toHaveTextContent('ADD ASSET');
  });

  test('Add Asset button fires onAddAsset callback', () => {
    const onAddAsset = jest.fn();
    renderSidebar({ onAddAsset });
    fireEvent.click(screen.getByTestId('add-asset-btn'));
    expect(onAddAsset).toHaveBeenCalledTimes(1);
  });

  test('active nav item has active styling class', () => {
    // Default route is / so DASHBOARD should be active
    renderSidebar();
    const dashboardLink = screen.getByTestId('nav-dashboard');
    expect(dashboardLink.className).toContain('bg-primary/10');
  });
});
