import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, TrendingUp, LineChart, Newspaper, Settings, Plus, Activity, Bitcoin } from 'lucide-react';

const Sidebar = ({ onAddAsset }) => {
  const location = useLocation();
  
  const navItems = [
    { path: '/', icon: LayoutDashboard, label: 'DASHBOARD', description: 'Portfolio overview' },
    { path: '/assets', icon: TrendingUp, label: 'ASSETS', description: 'All watchlist assets' },
    { path: '/backtest', icon: LineChart, label: 'BACKTEST LAB', description: 'Single-asset DCA backtest' },
    { path: '/simulation', icon: Activity, label: 'PORTFOLIO SIM', description: 'Multi-asset simulation' },
    { path: '/crypto-bot', icon: Bitcoin, label: 'CRYPTO BOT', description: 'Futures & grid trading' },
    { path: '/news', icon: Newspaper, label: 'NEWS & EVENTS', description: 'Market headlines' },
    { path: '/settings', icon: Settings, label: 'SETTINGS', description: 'Preferences' },
  ];

  const getIsActive = (path) => {
    if (path === '/') return location.pathname === '/';
    return location.pathname.startsWith(path);
  };
  
  return (
    <div className="fixed left-0 top-0 h-screen w-64 glass-effect border-r border-white/10 flex flex-col z-30" data-testid="sidebar">
      {/* Logo */}
      <div className="p-6 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-primary/20 flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight gold-text" data-testid="app-title">QUANTFI</h1>
            <p className="text-[10px] text-muted-foreground tracking-wider">DCA INTELLIGENCE PLATFORM</p>
          </div>
        </div>
      </div>
      
      {/* Navigation */}
      <nav className="flex-1 p-3 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = getIsActive(item.path);
          
          return (
            <Link
              key={item.path}
              to={item.path}
              data-testid={`nav-${item.label.toLowerCase().replace(/\s+/g, '-')}`}
              className={`flex items-center gap-3 px-4 py-3 rounded transition-all ${
                isActive
                  ? 'bg-primary/10 text-primary border border-primary/20'
                  : 'text-muted-foreground hover:text-foreground hover:bg-white/5'
              }`}
            >
              <Icon className={`w-5 h-5 ${isActive ? 'text-primary' : ''}`} />
              <div className="flex-1 min-w-0">
                <span className="text-sm font-medium block">{item.label}</span>
                {isActive && (
                  <span className="text-[10px] text-muted-foreground">{item.description}</span>
                )}
              </div>
              {isActive && <div className="w-1.5 h-1.5 rounded-full bg-primary" />}
            </Link>
          );
        })}
      </nav>
      
      {/* Add Asset Button */}
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
