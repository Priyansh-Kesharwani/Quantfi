import React from 'react';
import { ArrowUpRight, ArrowDownRight } from 'lucide-react';

/**
 * StatCard — KPI stat card with icon, value, sub-value, and optional trend.
 *
 * Extracted from Dashboard.js StatCard (lines 50-69).
 * Reusable across Dashboard, BacktestLab results, Signals summary,
 * and ValidationLab KPI rows.
 *
 * @param {React.ElementType} icon — Lucide icon component
 * @param {string} label — Uppercase label text
 * @param {string|number} value — Primary display value
 * @param {string} [subValue] — Secondary info text
 * @param {string} [trend] — Trend label text
 * @param {boolean} [trendPositive] — Whether trend is positive (green) or negative (red)
 * @param {string} [className] — Additional container classes
 */
const StatCard = ({ icon: Icon, label, value, subValue, trend, trendPositive, className = '' }) => (
  <div className={`glass-effect rounded-sm p-5 ${className}`} data-testid="stat-card">
    <div className="flex items-center gap-2 mb-3">
      {Icon && (
        <div className="p-2 rounded bg-white/5">
          <Icon className="w-4 h-4 text-primary" />
        </div>
      )}
      <span className="text-xs text-muted-foreground uppercase tracking-wider">{label}</span>
    </div>
    <div className="text-2xl font-bold font-data">{value}</div>
    <div className="flex items-center gap-2 mt-1">
      {subValue && <span className="text-xs text-muted-foreground font-data">{subValue}</span>}
      {trend != null && (
        <span className={`flex items-center gap-0.5 text-xs font-bold ${trendPositive ? 'text-emerald-400' : 'text-red-400'}`}>
          {trendPositive ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
          {trend}
        </span>
      )}
    </div>
  </div>
);

export default StatCard;
