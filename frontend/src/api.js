import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
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

  // Settings
  getSettings: () => axios.get(`${API}/settings`),
  updateSettings: (data) => axios.put(`${API}/settings`, data),

  // Dashboard
  getDashboard: () => axios.get(`${API}/dashboard`),

  // Health
  healthCheck: () => axios.get(`${API}/health`),
};

export default api;
