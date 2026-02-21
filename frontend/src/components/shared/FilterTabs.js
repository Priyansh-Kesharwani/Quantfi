import React from 'react';

/**
 * FilterTabs — Reusable filter tab bar.
 *
 * Replaces 2× identical patterns: category tabs in Assets page
 * and asset filter tabs in News page.
 *
 * @param {Array} options — [{key, label, color?}]
 * @param {string} value — Currently active key
 * @param {Function} onChange — Called with new key
 * @param {string} [className] — Additional classes
 * @param {string} [size='sm'] — 'sm' or 'md'
 */
const FilterTabs = ({ options = [], value, onChange, className = '', size = 'sm' }) => {
  const sizeClasses = size === 'md'
    ? 'px-3 py-1.5 text-xs'
    : 'px-2.5 py-1 text-[10px]';

  return (
    <div className={`flex items-center gap-1 flex-wrap ${className}`} data-testid="filter-tabs">
      {options.map(opt => {
        const isActive = opt.key === value;
        const activeClass = isActive
          ? opt.color
            ? `${opt.color} ring-1 ring-current`
            : 'bg-primary text-primary-foreground'
          : 'glass-effect hover:bg-white/10 text-muted-foreground';

        return (
          <button
            key={opt.key}
            onClick={() => onChange(opt.key)}
            className={`${sizeClasses} rounded font-bold transition ${activeClass}`}
            data-testid={`filter-tab-${opt.key}`}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
};

export default FilterTabs;
