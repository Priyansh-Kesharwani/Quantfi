import React, { useState, useEffect } from 'react';

import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

import api from '@/api';
import { Save, Settings as SettingsIcon, Sliders, Zap, Shield, Activity } from 'lucide-react';
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
    },
    phase_a: {
      entry_committee_method: 'trimmed_mean',
      entry_trim_pct: 0.1,
      exit_gamma_1: 0.4,
      exit_gamma_2: 0.35,
      exit_gamma_3: 0.25,
    },
    execution: {
      k_impact: 0.1,
      gamma: 0.5,
      sigma_slip: 0.001,
      mean_latency: 0.05,
      fee_bps: 10,
    },
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      const response = await api.getSettings();
      setSettings(prev => ({
        ...prev,
        ...response.data,
        score_weights: { ...prev.score_weights, ...response.data?.score_weights },
        phase_a: { ...prev.phase_a, ...response.data?.phase_a },
        execution: { ...prev.execution, ...response.data?.execution },
      }));
    } catch {
      console.warn('Settings endpoint not available');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await api.updateSettings(settings);
      toast.success('Settings saved successfully');
    } catch {
      toast.error('Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const updateWeight = (key, value) => {
    const numValue = parseFloat(value) / 100;
    setSettings({
      ...settings,
      score_weights: { ...settings.score_weights, [key]: numValue }
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
    <div className="p-6 lg:p-8 max-w-[1600px] mx-auto" data-testid="settings-page">
      <div className="mb-8">
        <h1 className="text-4xl font-bold tracking-tight mb-2" data-testid="settings-title">SETTINGS</h1>
        <p className="text-muted-foreground">Configure DCA parameters, scoring weights, and model settings</p>
      </div>

      <div className="max-w-4xl">
        <Tabs defaultValue="dca" className="w-full">
          <TabsList className="glass-effect mb-6 w-full justify-start" data-testid="settings-tabs">
            <TabsTrigger value="dca" className="font-bold text-xs">DCA DEFAULTS</TabsTrigger>
            <TabsTrigger value="weights" className="font-bold text-xs">SCORE WEIGHTS</TabsTrigger>
            <TabsTrigger value="phaseA" className="font-bold text-xs">PHASE A CONFIG</TabsTrigger>
            <TabsTrigger value="execution" className="font-bold text-xs">EXECUTION MODEL</TabsTrigger>
          </TabsList>

          <TabsContent value="dca">
            <div className="glass-effect rounded-sm p-6" data-testid="dca-defaults">
              <div className="flex items-center gap-2 mb-6"><Sliders className="w-5 h-5 text-primary" /><h2 className="text-xl font-bold">DCA DEFAULTS</h2></div>
              <div className="space-y-4">
                <div><label className="text-sm text-muted-foreground mb-2 block">DEFAULT DCA CADENCE</label><select value={settings.default_dca_cadence} onChange={(e) => setSettings({ ...settings, default_dca_cadence: e.target.value })} className="w-full p-3 glass-effect rounded text-sm font-data" data-testid="default-cadence-select"><option value="weekly">Weekly</option><option value="monthly">Monthly</option></select></div>
                <div><label className="text-sm text-muted-foreground mb-2 block">DEFAULT DCA AMOUNT (₹)</label><input type="number" value={settings.default_dca_amount} onChange={(e) => setSettings({ ...settings, default_dca_amount: parseFloat(e.target.value) })} className="w-full p-3 glass-effect rounded text-sm font-data" data-testid="default-amount-input" /></div>
                <div><label className="text-sm text-muted-foreground mb-2 block">DIP ALERT THRESHOLD</label><input type="number" value={settings.dip_alert_threshold} onChange={(e) => setSettings({ ...settings, dip_alert_threshold: parseFloat(e.target.value) })} className="w-full p-3 glass-effect rounded text-sm font-data" min="0" max="100" data-testid="dip-threshold-input" /><p className="text-xs text-muted-foreground mt-1">Get alerts when DCA score exceeds this threshold</p></div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="weights">
            <div className="glass-effect rounded-sm p-6" data-testid="score-weights">
              <div className="flex items-center gap-2 mb-2"><Activity className="w-5 h-5 text-primary" /><h2 className="text-xl font-bold">SCORE COMPONENT WEIGHTS</h2></div>
              <p className="text-sm text-muted-foreground mb-6">Total should equal 100%.</p>
              <div className="space-y-6">
                {[
                  { key: 'technical_momentum', label: 'TECHNICAL & MOMENTUM', desc: '200/50-day SMA, RSI, MACD, Bollinger Bands, ADX' },
                  { key: 'volatility_opportunity', label: 'VOLATILITY & OPPORTUNITY', desc: 'ATR percentile, drawdown from highs' },
                  { key: 'statistical_deviation', label: 'STATISTICAL DEVIATION', desc: 'Z-score (20, 50, 100-day windows)' },
                  { key: 'macro_fx', label: 'MACRO & FX', desc: 'USD-INR exchange rate vs historical average' },
                ].map(w => (
                  <div key={w.key}>
                    <div className="flex items-center justify-between mb-2"><label className="text-sm">{w.label}</label><span className="text-sm font-data font-bold">{(settings.score_weights[w.key] * 100).toFixed(0)}%</span></div>
                    <input type="range" min="0" max="100" value={settings.score_weights[w.key] * 100} onChange={(e) => updateWeight(w.key, e.target.value)} className="w-full" data-testid={`weight-${w.key.split('_')[0]}`} />
                    <p className="text-xs text-muted-foreground mt-1">{w.desc}</p>
                  </div>
                ))}
                <div className={`p-4 rounded ${Math.abs(totalWeight - 1.0) < 0.01 ? 'bg-chart-4/20 border border-chart-4/50' : 'bg-destructive/20 border border-destructive/50'}`}>
                  <div className="text-sm font-bold">TOTAL WEIGHT: {(totalWeight * 100).toFixed(0)}%</div>
                  {Math.abs(totalWeight - 1.0) >= 0.01 && <p className="text-xs mt-1">Warning: Total should equal 100%</p>}
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="phaseA">
            <div className="glass-effect rounded-sm p-6" data-testid="phase-a-config">
              <div className="flex items-center gap-2 mb-6"><Zap className="w-5 h-5 text-primary" /><h2 className="text-xl font-bold">PHASE A — ENTRY/EXIT SCORE CONFIG</h2></div>
              <p className="text-sm text-muted-foreground mb-6">Configures the composite score engine (OFI, Hawkes, LDC integration).</p>
              <div className="space-y-4">
                <div><label className="text-sm text-muted-foreground mb-2 block">ENTRY COMMITTEE METHOD</label><select value={settings.phase_a.entry_committee_method} onChange={(e) => setSettings({ ...settings, phase_a: { ...settings.phase_a, entry_committee_method: e.target.value } })} className="w-full p-3 glass-effect rounded text-sm font-data" data-testid="entry-method-select"><option value="trimmed_mean">Trimmed Mean</option><option value="mean">Simple Mean</option><option value="median">Median</option></select></div>
                <div><label className="text-sm text-muted-foreground mb-2 block">ENTRY TRIM PERCENTAGE</label><input type="number" value={settings.phase_a.entry_trim_pct} onChange={(e) => setSettings({ ...settings, phase_a: { ...settings.phase_a, entry_trim_pct: parseFloat(e.target.value) } })} className="w-full p-3 glass-effect rounded text-sm font-data" step="0.05" min="0" max="0.5" data-testid="entry-trim-input" /><p className="text-xs text-muted-foreground mt-1">Fraction of extreme values to trim (0-0.5)</p></div>
                <div className="border-t border-white/10 pt-4">
                  <h3 className="text-sm font-bold mb-4">EXIT SCORE WEIGHTS</h3>
                  <div className="grid grid-cols-3 gap-4">
                    <div><label className="text-[10px] text-muted-foreground block mb-1">γ₁ (TBL FLAG)</label><input type="number" value={settings.phase_a.exit_gamma_1} onChange={(e) => setSettings({ ...settings, phase_a: { ...settings.phase_a, exit_gamma_1: parseFloat(e.target.value) } })} className="w-full p-2 glass-effect rounded text-xs font-data" step="0.05" min="0" max="1" data-testid="exit-gamma1-input" /></div>
                    <div><label className="text-[10px] text-muted-foreground block mb-1">γ₂ (OFI REVERSAL)</label><input type="number" value={settings.phase_a.exit_gamma_2} onChange={(e) => setSettings({ ...settings, phase_a: { ...settings.phase_a, exit_gamma_2: parseFloat(e.target.value) } })} className="w-full p-2 glass-effect rounded text-xs font-data" step="0.05" min="0" max="1" data-testid="exit-gamma2-input" /></div>
                    <div><label className="text-[10px] text-muted-foreground block mb-1">γ₃ (λ DECAY)</label><input type="number" value={settings.phase_a.exit_gamma_3} onChange={(e) => setSettings({ ...settings, phase_a: { ...settings.phase_a, exit_gamma_3: parseFloat(e.target.value) } })} className="w-full p-2 glass-effect rounded text-xs font-data" step="0.05" min="0" max="1" data-testid="exit-gamma3-input" /></div>
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">Exit Score = γ₁·TBL + γ₂·OFI_reversal + γ₃·λ_decay. Should sum to 1.0</p>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="execution">
            <div className="glass-effect rounded-sm p-6" data-testid="execution-config">
              <div className="flex items-center gap-2 mb-6"><Shield className="w-5 h-5 text-primary" /><h2 className="text-xl font-bold">EXECUTION MODEL DEFAULTS</h2></div>
              <p className="text-sm text-muted-foreground mb-6">Phase 3 execution risk parameters used in backtesting and validation.</p>
              <div className="space-y-4">
                <div><label className="text-sm text-muted-foreground mb-2 block">IMPACT COEFFICIENT (k)</label><input type="number" value={settings.execution.k_impact} onChange={(e) => setSettings({ ...settings, execution: { ...settings.execution, k_impact: parseFloat(e.target.value) } })} className="w-full p-3 glass-effect rounded text-sm font-data" step="0.01" min="0" data-testid="exec-k-impact" /><p className="text-xs text-muted-foreground mt-1">Kyle model: impact = k · (order_size / ADV)^γ</p></div>
                <div><label className="text-sm text-muted-foreground mb-2 block">IMPACT GAMMA (γ)</label><input type="number" value={settings.execution.gamma} onChange={(e) => setSettings({ ...settings, execution: { ...settings.execution, gamma: parseFloat(e.target.value) } })} className="w-full p-3 glass-effect rounded text-sm font-data" step="0.1" min="0" max="2" data-testid="exec-gamma" /></div>
                <div><label className="text-sm text-muted-foreground mb-2 block">SLIPPAGE SIGMA (σ_slip)</label><input type="number" value={settings.execution.sigma_slip} onChange={(e) => setSettings({ ...settings, execution: { ...settings.execution, sigma_slip: parseFloat(e.target.value) } })} className="w-full p-3 glass-effect rounded text-sm font-data" step="0.0001" min="0" data-testid="exec-sigma-slip" /></div>
                <div><label className="text-sm text-muted-foreground mb-2 block">MEAN LATENCY (seconds)</label><input type="number" value={settings.execution.mean_latency} onChange={(e) => setSettings({ ...settings, execution: { ...settings.execution, mean_latency: parseFloat(e.target.value) } })} className="w-full p-3 glass-effect rounded text-sm font-data" step="0.01" min="0" data-testid="exec-latency" /></div>
                <div><label className="text-sm text-muted-foreground mb-2 block">TRANSACTION FEE (BPS)</label><input type="number" value={settings.execution.fee_bps} onChange={(e) => setSettings({ ...settings, execution: { ...settings.execution, fee_bps: parseFloat(e.target.value) } })} className="w-full p-3 glass-effect rounded text-sm font-data" min="0" data-testid="exec-fee-bps" /></div>
              </div>
            </div>
          </TabsContent>
        </Tabs>

        <button onClick={handleSave} disabled={saving || Math.abs(totalWeight - 1.0) >= 0.01}
          className="w-full mt-6 flex items-center justify-center gap-2 px-6 py-4 bg-primary text-primary-foreground rounded font-medium hover:bg-primary/90 transition disabled:opacity-50"
          data-testid="save-settings-btn">
          <Save className="w-5 h-5" /> {saving ? 'SAVING...' : 'SAVE SETTINGS'}
        </button>
      </div>
    </div>
  );
};

export default Settings;
