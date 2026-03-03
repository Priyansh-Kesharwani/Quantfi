import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from 'sonner';
import { WatchlistProvider } from './contexts/WatchlistContext';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Assets from './pages/Assets';
import AssetDetail from './pages/AssetDetail';
import BacktestLab from './pages/BacktestLab';
import News from './pages/News';
import Settings from './pages/Settings';
import PortfolioSim from './pages/PortfolioSim';
import CryptoBot from './pages/CryptoBot';
import PaperTrading from './pages/PaperTrading';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './components/ui/dialog';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import api from './api';
import { toast } from 'sonner';
import '@/index.css';

function App() {
  const [showAddAsset, setShowAddAsset] = useState(false);
  const [newAsset, setNewAsset] = useState({
    symbol: '',
    name: '',
    asset_type: 'equity',
    exchange: '',
    currency: 'USD'
  });
  const [adding, setAdding] = useState(false);

  useEffect(() => {
    const handleOpenAddAsset = () => setShowAddAsset(true);
    window.addEventListener('addAsset', handleOpenAddAsset);
    return () => window.removeEventListener('addAsset', handleOpenAddAsset);
  }, []);

  const handleAddAsset = async () => {
    const symbol = newAsset.symbol.trim();
    const name = newAsset.name.trim();

    if (!symbol) { toast.error('Symbol is required'); return; }
    if (!name) { toast.error('Asset name is required'); return; }
    if (symbol.length > 20) { toast.error('Symbol is too long'); return; }

    setAdding(true);
    try {
      const payload = { symbol, name, asset_type: newAsset.asset_type };
      if (newAsset.exchange) payload.exchange = newAsset.exchange;
      if (newAsset.currency && newAsset.currency !== 'USD') payload.currency = newAsset.currency;

      await api.addAsset(payload);
      toast.success(`${symbol} added to watchlist`);
      setShowAddAsset(false);
      setNewAsset({ symbol: '', name: '', asset_type: 'equity', exchange: '', currency: 'USD' });

      // Trigger global refresh so WatchlistContext re-fetches
      setTimeout(() => {
        window.dispatchEvent(new Event('refreshDashboard'));
      }, 500);
    } catch (error) {
      const errMsg = error?.response?.data?.detail || 'Failed to add asset. Check the symbol and try again.';
      toast.error(errMsg);
    } finally {
      setAdding(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <BrowserRouter>
        <WatchlistProvider>
          <div className="flex">
            <Sidebar onAddAsset={() => setShowAddAsset(true)} />
            
            <main className="flex-1 ml-64" data-testid="main-content">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/assets" element={<Assets />} />
                <Route path="/assets/:symbol" element={<AssetDetail />} />
                <Route path="/backtest" element={<BacktestLab />} />
                <Route path="/simulation" element={<PortfolioSim />} />
                <Route path="/crypto-bot" element={<CryptoBot />} />
                <Route path="/paper-trading" element={<PaperTrading />} />
                <Route path="/news" element={<News />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </main>
          </div>
        </WatchlistProvider>

        {/* Add Asset Dialog — high z-index so it appears above sidebar */}
        <Dialog open={showAddAsset} onOpenChange={setShowAddAsset}>
          <DialogContent className="glass-effect z-[100]" data-testid="add-asset-dialog">
            <DialogHeader>
              <DialogTitle className="text-2xl">ADD ASSET TO WATCHLIST</DialogTitle>
            </DialogHeader>
            
            <div className="space-y-4 mt-4">
              <div>
                <Label htmlFor="symbol" className="text-sm text-muted-foreground mb-2 block">
                  SYMBOL
                </Label>
                <Input
                  id="symbol"
                  placeholder="Yahoo Finance ticker symbol"
                  value={newAsset.symbol}
                  onChange={(e) => setNewAsset({ ...newAsset, symbol: e.target.value.toUpperCase() })}
                  className="glass-effect"
                  data-testid="asset-symbol-input"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Use the ticker symbol as listed on Yahoo Finance.
                </p>
              </div>

              <div>
                <Label htmlFor="name" className="text-sm text-muted-foreground mb-2 block">
                  NAME
                </Label>
                <Input
                  id="name"
                  placeholder="e.g., Gold Futures, Apple Inc."
                  value={newAsset.name}
                  onChange={(e) => setNewAsset({ ...newAsset, name: e.target.value })}
                  className="glass-effect"
                  data-testid="asset-name-input"
                />
              </div>

              <div>
                <Label htmlFor="type" className="text-sm text-muted-foreground mb-2 block">
                  ASSET TYPE
                </Label>
                <Select
                  value={newAsset.asset_type}
                  onValueChange={(value) => setNewAsset({ ...newAsset, asset_type: value })}
                >
                  <SelectTrigger className="glass-effect" data-testid="asset-type-select">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="equity">Equity</SelectItem>
                    <SelectItem value="etf">ETF</SelectItem>
                    <SelectItem value="commodity">Commodity / Futures</SelectItem>
                    <SelectItem value="crypto">Cryptocurrency</SelectItem>
                    <SelectItem value="index">Index</SelectItem>
                    <SelectItem value="other">Other</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="exchange" className="text-sm text-muted-foreground mb-2 block">
                  EXCHANGE (OPTIONAL)
                </Label>
                <Input
                  id="exchange"
                  placeholder="Leave blank for auto-detect"
                  value={newAsset.exchange}
                  onChange={(e) => setNewAsset({ ...newAsset, exchange: e.target.value.toUpperCase() })}
                  className="glass-effect"
                  data-testid="asset-exchange-select"
                />
              </div>

              <Button
                onClick={handleAddAsset}
                disabled={adding}
                className="w-full bg-primary text-primary-foreground hover:bg-primary/90"
                data-testid="confirm-add-asset-btn"
              >
                {adding ? 'ADDING...' : 'ADD ASSET'}
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </BrowserRouter>
      
      <Toaster position="top-right" theme="dark" />
    </div>
  );
}

export default App;
