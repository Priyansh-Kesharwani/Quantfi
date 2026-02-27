import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
const API = `${BACKEND_URL}/api`;

export const api = {
  // ── Assets ──────────────────────────────────────────────
  getAssets: () => axios.get(`${API}/assets`),
  addAsset: (data) => axios.post(`${API}/assets`, data),
  removeAsset: (symbol) => axios.delete(`${API}/assets/${symbol}`),

  // ── Prices ──────────────────────────────────────────────
  getLatestPrice: (symbol) => axios.get(`${API}/prices/${symbol}`),
  getPriceHistory: (symbol, period = '1y') => axios.get(`${API}/prices/${symbol}/history?period=${period}`),

  // ── Indicators (Phase 0 — basic) ───────────────────────
  getIndicators: (symbol) => axios.get(`${API}/indicators/${symbol}`),

  // ── Scores (Phase 0 — DCA composite) ───────────────────
  getScore: (symbol) => axios.get(`${API}/scores/${symbol}`),

  // ── Backtest ────────────────────────────────────────────
  runBacktest: (data) => axios.post(`${API}/backtest`, data),
  runEnhancedBacktest: (data) => axios.post(`${API}/backtest/enhanced`, data),

  // ── News ────────────────────────────────────────────────
  getNews: (limit = 20) => axios.get(`${API}/news?limit=${limit}`),
  refreshNews: () => axios.post(`${API}/news/refresh`),
  getNewsForAsset: (symbol, limit = 2) => axios.get(`${API}/news/asset/${symbol}?limit=${limit}`),

  // ── Sentiment ───────────────────────────────────────────
  getSentiment: (symbol) => axios.get(`${API}/sentiment/${symbol}`),
  runSentiment: (symbol) => axios.post(`${API}/sentiment/${symbol}`),

  // ── Settings ────────────────────────────────────────────
  getSettings: () => axios.get(`${API}/settings`),
  updateSettings: (data) => axios.put(`${API}/settings`, data),

  // ── Dashboard ───────────────────────────────────────────
  getDashboard: () => axios.get(`${API}/dashboard`),

  // ── Health ──────────────────────────────────────────────
  healthCheck: () => axios.get(`${API}/health`),

  // ── Phase 1 Advanced Indicators (promoted from /dev/) ──
  getPhase1Indicators: (symbol) => axios.get(`${API}/phase1/indicators/${symbol}`),
  getPhase1Health: () => axios.get(`${API}/phase1/health`),

  // ── Phase A: Entry/Exit Scores & Microstructure ────────
  getPhaseAScores: (symbol) => axios.get(`${API}/phaseA/scores/${symbol}`),
  getMicrostructure: (symbol) => axios.get(`${API}/phaseA/microstructure/${symbol}`),

  // ── Signals ─────────────────────────────────────────────
  getSignalsSummary: () => axios.get(`${API}/signals/summary`),
  getActiveSignals: () => axios.get(`${API}/signals/active`),
  getSignalHistory: (symbol, period = '3m') => axios.get(`${API}/signals/${symbol}/history?period=${period}`),
  getSignalTrades: (symbol) => axios.get(`${API}/signals/${symbol}/trades`),
  getSignalPerformance: (symbol) => axios.get(`${API}/signals/${symbol}/performance`),

  // ── Portfolio Simulation ───────────────────────────────────
  runSimulation: (data) => axios.post(`${API}/simulation/run`, data),
  getSimulationTemplates: () => axios.get(`${API}/simulation/templates`),
  getSimulationCostPresets: () => axios.get(`${API}/simulation/cost-presets`),

  // ── Crypto Bot ──────────────────────────────────────────────
  runCryptoBacktest: (data) => axios.post(`${API}/crypto/backtest`, data),
  getCryptoDefaults: () => axios.get(`${API}/crypto/config/defaults`),
  getCryptoStrategies: () => axios.get(`${API}/crypto/strategies`),
  getCryptoMarkets: (exchange = 'binance') => axios.get(`${API}/crypto/markets?exchange=${exchange}`),
};

export default api;
