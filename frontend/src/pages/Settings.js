import React, { useState, useEffect } from 'react';
import api from '../api';
import { Save, Settings as SettingsIcon } from 'lucide-react';
import { toast } from 'sonner';

const Settings = () => {
  const [settings, setSettings] = useState({
    default_dca_cadence: 'monthly',
    default_dca_amount: 5000,
    dip_alert_threshold: 70,
    score_weights: {
      technical_momentum: 0.4,
      volatility_opportunity: 0.2,
      statistical_deviation: 0.2,
      macro_fx: 0.2
    }
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      const response = await api.getSettings();
      setSettings(response.data);
    } catch (error) {
      console.error('Error fetching settings:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await api.updateSettings(settings);
      toast.success('Settings saved successfully');
    } catch (error) {
      console.error('Error saving settings:', error);
      toast.error('Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const updateWeight = (key, value) => {
    const numValue = parseFloat(value) / 100;
    setSettings({
      ...settings,
      score_weights: {
        ...settings.score_weights,
        [key]: numValue
      }
    });
  };

  const totalWeight = Object.values(settings.score_weights).reduce((sum, val) => sum + val, 0);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <SettingsIcon className="w-12 h-12 animate-pulse mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Loading settings...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-8" data-testid="settings-page">
      <div className="mb-8">
        <h1 className="text-4xl font-bold tracking-tight mb-2" data-testid="settings-title">SETTINGS</h1>
        <p className="text-muted-foreground">Configure DCA parameters and scoring weights</p>
      </div>

      <div className="max-w-4xl space-y-6">
        {/* DCA Defaults */}
        <div className="glass-effect rounded-sm p-6" data-testid="dca-defaults">
          <h2 className="text-xl font-bold mb-6">DCA DEFAULTS</h2>
          
          <div className="space-y-4">
            <div>
              <label className="text-sm text-muted-foreground mb-2 block">DEFAULT DCA CADENCE</label>
              <select
                value={settings.default_dca_cadence}
                onChange={(e) => setSettings({ ...settings, default_dca_cadence: e.target.value })}
                className="w-full p-3 glass-effect rounded text-sm font-data"
                data-testid="default-cadence-select"
              >
                <option value="weekly">Weekly</option>
                <option value="monthly">Monthly</option>
              </select>
            </div>

            <div>
              <label className="text-sm text-muted-foreground mb-2 block">DEFAULT DCA AMOUNT (₹)</label>
              <input
                type="number"
                value={settings.default_dca_amount}
                onChange={(e) => setSettings({ ...settings, default_dca_amount: parseFloat(e.target.value) })}
                className="w-full p-3 glass-effect rounded text-sm font-data"
                data-testid="default-amount-input"
              />
            </div>

            <div>
              <label className="text-sm text-muted-foreground mb-2 block">DIP ALERT THRESHOLD</label>
              <input
                type="number"
                value={settings.dip_alert_threshold}
                onChange={(e) => setSettings({ ...settings, dip_alert_threshold: parseFloat(e.target.value) })}
                className="w-full p-3 glass-effect rounded text-sm font-data"
                min="0"
                max="100"
                data-testid="dip-threshold-input"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Get alerts when DCA score exceeds this threshold
              </p>
            </div>
          </div>
        </div>

        {/* Score Weights */}
        <div className="glass-effect rounded-sm p-6" data-testid="score-weights">
          <h2 className="text-xl font-bold mb-2">SCORE COMPONENT WEIGHTS</h2>
          <p className="text-sm text-muted-foreground mb-6">
            Adjust how each factor contributes to the composite DCA score. Total should equal 100%.
          </p>
          
          <div className="space-y-6">
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm">TECHNICAL & MOMENTUM</label>
                <span className="text-sm font-data font-bold">
                  {(settings.score_weights.technical_momentum * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={settings.score_weights.technical_momentum * 100}
                onChange={(e) => updateWeight('technical_momentum', e.target.value)}
                className="w-full"
                data-testid="weight-technical"
              />
              <p className="text-xs text-muted-foreground mt-1">
                200/50-day SMA, RSI, MACD, Bollinger Bands, ADX
              </p>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm">VOLATILITY & OPPORTUNITY</label>
                <span className="text-sm font-data font-bold">
                  {(settings.score_weights.volatility_opportunity * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={settings.score_weights.volatility_opportunity * 100}
                onChange={(e) => updateWeight('volatility_opportunity', e.target.value)}
                className="w-full"
                data-testid="weight-volatility"
              />
              <p className="text-xs text-muted-foreground mt-1">
                ATR percentile, drawdown from highs
              </p>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm">STATISTICAL DEVIATION</label>
                <span className="text-sm font-data font-bold">
                  {(settings.score_weights.statistical_deviation * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={settings.score_weights.statistical_deviation * 100}
                onChange={(e) => updateWeight('statistical_deviation', e.target.value)}
                className="w-full"
                data-testid="weight-statistical"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Z-score (20, 50, 100-day windows)
              </p>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm">MACRO & FX</label>
                <span className="text-sm font-data font-bold">
                  {(settings.score_weights.macro_fx * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={settings.score_weights.macro_fx * 100}
                onChange={(e) => updateWeight('macro_fx', e.target.value)}
                className="w-full"
                data-testid="weight-macro"
              />
              <p className="text-xs text-muted-foreground mt-1">
                USD-INR exchange rate vs historical average
              </p>
            </div>

            {/* Total Weight Indicator */}
            <div className={`p-4 rounded ${
              Math.abs(totalWeight - 1.0) < 0.01 ? 'bg-chart-4/20 border border-chart-4/50' : 'bg-destructive/20 border border-destructive/50'
            }`}>
              <div className="text-sm font-bold">
                TOTAL WEIGHT: {(totalWeight * 100).toFixed(0)}%
              </div>
              {Math.abs(totalWeight - 1.0) >= 0.01 && (
                <p className="text-xs mt-1">Warning: Total should equal 100%</p>
              )}
            </div>
          </div>
        </div>

        {/* Save Button */}
        <button
          onClick={handleSave}
          disabled={saving || Math.abs(totalWeight - 1.0) >= 0.01}
          className="w-full flex items-center justify-center gap-2 px-6 py-4 bg-primary text-primary-foreground rounded font-medium hover:bg-primary/90 transition disabled:opacity-50"
          data-testid="save-settings-btn"
        >
          <Save className="w-5 h-5" />
          {saving ? 'SAVING...' : 'SAVE SETTINGS'}
        </button>
      </div>
    </div>
  );
};

export default Settings;
