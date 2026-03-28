import React from 'react';

/**
 * AssetPicker — Dropdown selector for choosing an asset from the watchlist.
 *
 * Replaces inline <select> in BacktestLab and will be reused in
 * Signals, ValidationLab, and any future pages that need asset selection.
 *
 * @param {Array} assets — Array of asset objects [{symbol, name, asset_type}]
 * @param {string} value — Currently selected symbol
 * @param {Function} onChange — Called with new symbol
 * @param {string} [label] — Optional label above the picker
 * @param {string} [className] — Additional classes
 * @param {string} [testId] — data-testid
 */
const AssetPicker = ({ assets = [], value, onChange, label, className = '', testId = 'asset-picker' }) => (
  <div className={className}>
    {label && (
      <label className="text-sm text-muted-foreground mb-2 block">{label}</label>
    )}
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full p-3 glass-effect rounded text-sm font-data"
      data-testid={testId}
    >
      {assets.length === 0 && (
        <option value="">No assets — add to watchlist first</option>
      )}
      {assets.map(asset => (
        <option key={asset.symbol} value={asset.symbol}>
          {asset.symbol} — {asset.name}
        </option>
      ))}
    </select>
  </div>
);

export default AssetPicker;
