export const formatCurrency = (value, currency = 'USD') => {
  if (!value && value !== 0) return '-';
  
  if (currency === 'INR') {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 2
    }).format(value);
  }
  
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2
  }).format(value);
};

export const formatNumber = (value, decimals = 2) => {
  if (!value && value !== 0) return '-';
  return new Intl.NumberFormat('en-US', {
    maximumFractionDigits: decimals,
    minimumFractionDigits: decimals
  }).format(value);
};

export const formatPercent = (value, decimals = 2) => {
  if (!value && value !== 0) return '-';
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(decimals)}%`;
};

export const getScoreColor = (score) => {
  if (score >= 81) return '#22C55E';
  if (score >= 61) return '#10B981';
  if (score >= 31) return '#F59E0B';
  return '#EF4444';
};

export const getZoneLabel = (zone) => {
  const labels = {
    'strong_buy': 'STRONG BUY DIP',
    'favorable': 'FAVORABLE',
    'neutral': 'NEUTRAL',
    'unfavorable': 'UNFAVORABLE'
  };
  return labels[zone] || zone.toUpperCase();
};

export const getAssetSymbol = (symbol) => {
  const map = {
    'GOLD': 'XAU',
    'SILVER': 'XAG',
    'GC=F': 'GOLD',
    'SI=F': 'SILVER'
  };
  return map[symbol] || symbol;
};
