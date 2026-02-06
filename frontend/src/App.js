import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from 'sonner';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import BacktestLab from './pages/BacktestLab';
import News from './pages/News';
import Settings from './pages/Settings';
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
    asset_type: 'equity'
  });
  const [adding, setAdding] = useState(false);

  useEffect(() => {
    // Listen for add asset event
    const handleAddAsset = () => setShowAddAsset(true);
    window.addEventListener('addAsset', handleAddAsset);
    return () => window.removeEventListener('addAsset', handleAddAsset);
  }, []);

  const handleAddAsset = async () => {
    if (!newAsset.symbol || !newAsset.name) {
      toast.error('Please fill all fields');
      return;
    }

    setAdding(true);
    try {
      await api.addAsset(newAsset);
      toast.success(`${newAsset.symbol} added successfully`);
      setShowAddAsset(false);
      setNewAsset({ symbol: '', name: '', asset_type: 'equity' });
      // Refresh the page to show new asset
      window.location.reload();
    } catch (error) {
      console.error('Error adding asset:', error);
      toast.error('Failed to add asset');
    } finally {
      setAdding(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <BrowserRouter>
        <div className="flex">
          <Sidebar onAddAsset={() => setShowAddAsset(true)} />
          
          <main className="flex-1 ml-64" data-testid="main-content">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/backtest" element={<BacktestLab />} />
              <Route path="/news" element={<News />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </main>
        </div>

        {/* Add Asset Dialog */}
        <Dialog open={showAddAsset} onOpenChange={setShowAddAsset}>
          <DialogContent className="glass-effect" data-testid="add-asset-dialog">
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
                  placeholder="e.g., GOLD, SILVER, AAPL, TSLA"
                  value={newAsset.symbol}
                  onChange={(e) => setNewAsset({ ...newAsset, symbol: e.target.value.toUpperCase() })}
                  className="glass-effect"
                  data-testid="asset-symbol-input"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  For metals: GOLD or SILVER. For US stocks: use ticker symbol.
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
                    <SelectItem value="metal">Metal (Gold/Silver)</SelectItem>
                    <SelectItem value="equity">US Equity</SelectItem>
                  </SelectContent>
                </Select>
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
