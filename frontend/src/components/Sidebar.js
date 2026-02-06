import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Home, TrendingUp, LineChart, Newspaper, Settings, Plus } from 'lucide-react';

const Sidebar = ({ onAddAsset }) => {
  const location = useLocation();
  
  const navItems = [
    { path: '/', icon: Home, label: 'DASHBOARD' },
    { path: '/assets', icon: TrendingUp, label: 'ASSETS' },
    { path: '/backtest', icon: LineChart, label: 'BACKTEST LAB' },
    { path: '/news', icon: Newspaper, label: 'NEWS & EVENTS' },
    { path: '/settings', icon: Settings, label: 'SETTINGS' },
  ];
  
  return (
    <div className="fixed left-0 top-0 h-screen w-64 glass-effect border-r border-white/10 flex flex-col" data-testid="sidebar">
      <div className="p-6 border-b border-white/10">
        <h1 className="text-2xl font-bold tracking-tight gold-text" data-testid="app-title">QUANTFI</h1>
        <p className="text-xs text-muted-foreground mt-1">DCA Intelligence Platform</p>
      </div>
      
      <nav className="flex-1 p-4 space-y-2">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.path;
          
          return (
            <Link
              key={item.path}
              to={item.path}
              data-testid={`nav-${item.label.toLowerCase().replace(/\s+/g, '-')}`}
              className={`flex items-center gap-3 px-4 py-3 rounded text-sm transition-all ${
                isActive
                  ? 'bg-primary/10 text-primary border border-primary/20'
                  : 'text-muted-foreground hover:text-foreground hover:bg-white/5'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
            </Link>
          );
        })}
      </nav>
      
      <div className="p-4 border-t border-white/10">
        <button
          onClick={onAddAsset}
          data-testid="add-asset-btn"
          className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary hover:bg-primary/90 text-primary-foreground rounded font-medium text-sm transition-colors"
        >
          <Plus className="w-5 h-5" />
          ADD ASSET
        </button>
      </div>
    </div>
  );
};

export default Sidebar;
