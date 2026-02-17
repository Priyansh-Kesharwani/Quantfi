import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
const API = `${BACKEND_URL}/api`;

export const api = {
  // Assets
  getAssets: () => axios.get(`${API}/assets`),
  addAsset: (data) => axios.post(`${API}/assets`, data),
  removeAsset: (symbol) => axios.delete(`${API}/assets/${symbol}`),

  // Prices
  getLatestPrice: (symbol) => axios.get(`${API}/prices/${symbol}`),
  getPriceHistory: (symbol, period = '1y') => axios.get(`${API}/prices/${symbol}/history?period=${period}`),

  // Indicators
  getIndicators: (symbol) => axios.get(`${API}/indicators/${symbol}`),

  // Scores
  getScore: (symbol) => axios.get(`${API}/scores/${symbol}`),

  // Backtest
  runBacktest: (data) => axios.post(`${API}/backtest`, data),

  // News
  getNews: (limit = 20) => axios.get(`${API}/news?limit=${limit}`),
  refreshNews: () => axios.post(`${API}/news/refresh`),
  // Per-asset news headlines — TODO Phase 2: backend endpoint
  getNewsForAsset: (symbol, limit = 2) => axios.get(`${API}/news/asset/${symbol}?limit=${limit}`),

  // Sentiment
  getSentiment: (symbol) => axios.get(`${API}/sentiment/${symbol}`),
  runSentiment: (symbol) => axios.post(`${API}/sentiment/${symbol}`),

  // Settings
  getSettings: () => axios.get(`${API}/settings`),
  updateSettings: (data) => axios.put(`${API}/settings`, data),

  // Dashboard
  getDashboard: () => axios.get(`${API}/dashboard`),

  // Health
  healthCheck: () => axios.get(`${API}/health`),
};

export default api;
