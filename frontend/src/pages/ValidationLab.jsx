import React, { useState, useEffect, useMemo } from 'react';

import { PageShell, AssetPicker, StatCard, MetricGrid, RefreshButton } from '@/components/shared';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

import { useWatchlist } from '@/contexts/WatchlistContext';

import api from '@/api';
import { toast } from 'sonner';
import {
  FlaskConical, Play, CheckCircle, XCircle, AlertTriangle,
  TrendingUp, Activity, BarChart3, Zap
} from 'lucide-react';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell, CartesianGrid
} from 'recharts';

const TuningTab = ({ data }) => {
  if (!data) {
    return (
      <div className="glass-effect rounded-sm p-12 text-center">
        <FlaskConical className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
        <h3 className="text-xl font-bold mb-2">NO TUNING DATA</h3>
        <p className="text-muted-foreground text-sm">Run Phase 3 validation to generate tuning results.</p>
      </div>
    );
  }

  const bestParams = data.best_params || {};
  const ablation = data.ablation || [];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={Zap} label="Best S(θ)" value={(data.best_score ?? 0).toFixed(2)} />
        <StatCard icon={TrendingUp} label="OOS Sortino" value={(data.oos_sortino ?? 0).toFixed(2)} />
        <StatCard icon={BarChart3} label="OOS CAGR" value={`${((data.oos_cagr ?? 0) * 100).toFixed(1)}%`} />
        <StatCard icon={Activity} label="Method" value={data.method || 'Bayesian'} />
      </div>
      <div className="glass-effect rounded-sm p-6">
        <h3 className="text-lg font-bold mb-4">BEST PARAMETERS FOUND</h3>
        <MetricGrid items={Object.entries(bestParams).map(([k, v]) => ({ label: k, value: typeof v === 'number' ? v.toFixed(4) : String(v) }))} cols={4} />
        <div className="mt-4 pt-4 border-t border-white/10 text-xs text-muted-foreground">Objective: S(θ) = median(M) − λ_var·std(M) = {(data.best_score ?? 0).toFixed(4)}</div>
      </div>
      {ablation.length > 0 && (
        <div className="glass-effect rounded-sm p-6">
          <h3 className="text-lg font-bold mb-4">ABLATION STUDY</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead><tr className="border-b border-white/10"><th className="text-left text-[10px] text-muted-foreground py-2 uppercase">Component</th><th className="text-right text-[10px] text-muted-foreground py-2 uppercase">Removed → ΔMetric</th><th className="text-right text-[10px] text-muted-foreground py-2 uppercase">Importance</th></tr></thead>
              <tbody>{ablation.map((row, i) => (
                <tr key={i} className="border-b border-white/[0.04]"><td className="py-3 text-sm font-bold">{row.component}</td><td className="py-3 text-right text-sm font-data text-red-400">{(row.delta ?? 0).toFixed(3)}</td><td className="py-3 text-right"><div className="flex items-center justify-end gap-2"><div className="w-24 bg-white/5 rounded-full h-2 overflow-hidden"><div className="h-full bg-primary rounded-full" style={{ width: `${Math.min(100, (row.importance ?? 0) * 100)}%` }} /></div><span className="text-xs font-data">{((row.importance ?? 0) * 100).toFixed(0)}%</span></div></td></tr>
              ))}</tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

const WalkForwardTab = ({ data }) => {
  if (!data) {
    return (<div className="glass-effect rounded-sm p-12 text-center"><FlaskConical className="w-16 h-16 mx-auto mb-4 text-muted-foreground" /><h3 className="text-xl font-bold mb-2">NO WALK-FORWARD DATA</h3><p className="text-muted-foreground text-sm">Run validation to generate walk-forward results.</p></div>);
  }
  const folds = data.folds || [];
  const stability = data.stability_std ?? null;
  const stableOk = stability !== null && stability < 0.5;

  return (
    <div className="space-y-6">
      <div className="glass-effect rounded-sm p-6">
        <h3 className="text-lg font-bold mb-4">WALK-FORWARD FOLDS</h3>
        <div className="space-y-2">{folds.map((fold, i) => (
          <div key={i} className="flex items-center gap-3"><span className="text-xs font-data text-muted-foreground w-12">Fold {i + 1}</span><div className="flex-1 flex h-6 rounded overflow-hidden"><div className="bg-primary/30 flex items-center justify-center text-[9px] font-data" style={{ width: `${(fold.train_pct ?? 75)}%` }}>TRAIN {fold.train_period || ''}</div><div className="bg-chart-4/40 flex items-center justify-center text-[9px] font-data" style={{ width: `${(fold.test_pct ?? 25)}%` }}>TEST {fold.test_period || ''}</div></div></div>
        ))}</div>
      </div>
      {folds.length > 0 && (
        <div className="glass-effect rounded-sm p-6">
          <h3 className="text-lg font-bold mb-4">PER-FOLD METRICS</h3>
          <div className="overflow-x-auto">
            <table className="w-full"><thead><tr className="border-b border-white/10"><th className="text-left text-[10px] text-muted-foreground py-2 uppercase">Fold</th><th className="text-right text-[10px] text-muted-foreground py-2 uppercase">OOS Sortino</th><th className="text-right text-[10px] text-muted-foreground py-2 uppercase">OOS IC</th><th className="text-right text-[10px] text-muted-foreground py-2 uppercase">OOS CAGR</th></tr></thead>
            <tbody>{folds.map((fold, i) => (<tr key={i} className="border-b border-white/[0.04]"><td className="py-3 text-sm font-bold">Fold {i + 1}</td><td className="py-3 text-right font-data text-sm">{(fold.oos_sortino ?? 0).toFixed(2)}</td><td className="py-3 text-right font-data text-sm">{(fold.oos_ic ?? 0).toFixed(3)}</td><td className="py-3 text-right font-data text-sm">{((fold.oos_cagr ?? 0) * 100).toFixed(1)}%</td></tr>))}</tbody>
            </table>
          </div>
          {stability !== null && <div className={`mt-4 pt-4 border-t border-white/10 flex items-center gap-2 text-sm ${stableOk ? 'text-emerald-400' : 'text-red-400'}`}>{stableOk ? <CheckCircle className="w-4 h-4" /> : <XCircle className="w-4 h-4" />}Stability (std across folds): {stability.toFixed(3)} {stableOk ? '✓ Below 0.5 threshold' : '✗ Exceeds 0.5 threshold'}</div>}
        </div>
      )}
    </div>
  );
};

const HawkesStressTab = ({ data }) => {
  if (!data) return (<div className="glass-effect rounded-sm p-12 text-center"><FlaskConical className="w-16 h-16 mx-auto mb-4 text-muted-foreground" /><h3 className="text-xl font-bold mb-2">NO HAWKES STRESS DATA</h3><p className="text-muted-foreground text-sm">Run validation to generate Hawkes stress test results.</p></div>);
  const regimes = data.regimes || [];
  return (
    <div className="space-y-6"><div className="glass-effect rounded-sm p-6"><h3 className="text-lg font-bold mb-4">HAWKES REGIME SIMULATION RESULTS</h3><div className="overflow-x-auto"><table className="w-full"><thead><tr className="border-b border-white/10"><th className="text-left text-[10px] text-muted-foreground py-2 uppercase">Regime</th><th className="text-right text-[10px] text-muted-foreground py-2 uppercase">μ</th><th className="text-right text-[10px] text-muted-foreground py-2 uppercase">α</th><th className="text-right text-[10px] text-muted-foreground py-2 uppercase">β</th><th className="text-right text-[10px] text-muted-foreground py-2 uppercase">η</th><th className="text-right text-[10px] text-muted-foreground py-2 uppercase">Events</th><th className="text-right text-[10px] text-muted-foreground py-2 uppercase">RMSE</th><th className="text-center text-[10px] text-muted-foreground py-2 uppercase">Pass</th></tr></thead><tbody>{regimes.map((r, i) => (<tr key={i} className="border-b border-white/[0.04]"><td className="py-3 text-sm font-bold">{r.name}</td><td className="py-3 text-right font-data text-sm">{(r.mu ?? 0).toFixed(2)}</td><td className="py-3 text-right font-data text-sm">{(r.alpha ?? 0).toFixed(2)}</td><td className="py-3 text-right font-data text-sm">{(r.beta ?? 0).toFixed(2)}</td><td className="py-3 text-right font-data text-sm">{(r.eta ?? 0).toFixed(2)}</td><td className="py-3 text-right font-data text-sm">{r.event_count ?? 0}</td><td className="py-3 text-right font-data text-sm">{((r.rmse ?? 0) * 100).toFixed(1)}%</td><td className="py-3 text-center">{r.passed ? <CheckCircle className="w-4 h-4 text-emerald-400 inline" /> : <XCircle className="w-4 h-4 text-red-400 inline" />}</td></tr>))}</tbody></table></div></div></div>
  );
};

const ExecRiskTab = ({ data }) => {
  if (!data) return (<div className="glass-effect rounded-sm p-12 text-center"><FlaskConical className="w-16 h-16 mx-auto mb-4 text-muted-foreground" /><h3 className="text-xl font-bold mb-2">NO EXECUTION RISK DATA</h3><p className="text-muted-foreground text-sm">Run validation to generate execution risk analysis.</p></div>);
  const breakdown = data.pnl_breakdown || {};
  const barData = Object.entries(breakdown).map(([k, v]) => ({ name: k.charAt(0).toUpperCase() + k.slice(1), value: Math.abs(v * 100), pct: v * 100, fill: '#EF4444' }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={Activity} label="Avg Slippage" value={`${((data.avg_slippage ?? 0) * 100).toFixed(2)}%`} />
        <StatCard icon={BarChart3} label="Fill Rate" value={`${((data.fill_rate ?? 0) * 100).toFixed(1)}%`} />
        <StatCard icon={Zap} label="Avg Impact" value={`${((data.avg_impact ?? 0) * 10000).toFixed(1)}bp`} />
        <StatCard icon={TrendingUp} label="Total Erosion" value={`${((data.total_erosion ?? 0) * 100).toFixed(2)}%`} />
      </div>
      {barData.length > 0 && (
        <div className="glass-effect rounded-sm p-6"><h3 className="text-lg font-bold mb-4">PnL EROSION BY COMPONENT</h3><div style={{ width: '100%', height: 220 }}><ResponsiveContainer><BarChart data={barData} layout="vertical" margin={{ top: 5, right: 30, left: 80, bottom: 5 }}><CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" /><XAxis type="number" tick={{ fontSize: 10, fill: '#888' }} /><YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: '#888' }} /><Tooltip content={({ active, payload }) => { if (!active || !payload?.length) return null; return (<div className="glass-effect rounded p-2 text-xs border border-white/10"><p className="font-bold">{payload[0].payload.name}</p><p className="text-red-400">{payload[0].payload.pct.toFixed(3)}%</p></div>); }} /><Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>{barData.map((_, i) => <Cell key={i} fill="#EF4444" fillOpacity={0.7} />)}</Bar></BarChart></ResponsiveContainer></div></div>
      )}
      <div className="glass-effect rounded-sm p-6 border-l-4 border-primary"><h3 className="font-bold mb-2">FILL MODEL</h3><p className="text-sm text-muted-foreground font-data">fill_price = mid + sign · impact(queue_pos, volume) + N(0, σ_slip)</p></div>
    </div>
  );
};

const ValidationLab = () => {
  const { assetList, loading: wlLoading } = useWatchlist();
  const [selectedAsset, setSelectedAsset] = useState('');
  const [running, setRunning] = useState(false);
  const [tuningData, setTuningData] = useState(null);
  const [walkforwardData, setWalkforwardData] = useState(null);
  const [hawkesData, setHawkesData] = useState(null);
  const [execRiskData, setExecRiskData] = useState(null);

  useEffect(() => { if (assetList.length > 0 && !selectedAsset) setSelectedAsset(assetList[0].symbol); }, [assetList, selectedAsset]);

  useEffect(() => {
    if (!selectedAsset) return;
    const fetchAll = async () => {
      const [t, w, h, e] = await Promise.allSettled([api.getValidationTuning(selectedAsset), api.getValidationWalkforward(selectedAsset), api.getValidationHawkes(selectedAsset), api.getValidationExecRisk(selectedAsset)]);
      if (t.status === 'fulfilled') setTuningData(t.value.data);
      if (w.status === 'fulfilled') setWalkforwardData(w.value.data);
      if (h.status === 'fulfilled') setHawkesData(h.value.data);
      if (e.status === 'fulfilled') setExecRiskData(e.value.data);
    };
    fetchAll().catch(() => {});
  }, [selectedAsset]);

  const handleRun = async () => {
    if (!selectedAsset) return;
    setRunning(true);
    try {
      await api.runValidation(selectedAsset, {});
      toast.success('Validation run complete');
      const [t, w, h, e] = await Promise.allSettled([api.getValidationTuning(selectedAsset), api.getValidationWalkforward(selectedAsset), api.getValidationHawkes(selectedAsset), api.getValidationExecRisk(selectedAsset)]);
      if (t.status === 'fulfilled') setTuningData(t.value.data);
      if (w.status === 'fulfilled') setWalkforwardData(w.value.data);
      if (h.status === 'fulfilled') setHawkesData(h.value.data);
      if (e.status === 'fulfilled') setExecRiskData(e.value.data);
    } catch (err) { toast.error(err?.response?.data?.detail || 'Validation run failed'); } finally { setRunning(false); }
  };

  return (
    <PageShell title="VALIDATION LAB" subtitle="Phase 3 model validation, tuning, and stress testing" loading={wlLoading}
      isEmpty={assetList.length === 0} emptyTitle="NO ASSETS TO VALIDATE" emptyMessage="Add assets to your watchlist first, then run validation."
      onEmpty={() => window.dispatchEvent(new Event('addAsset'))} testId="validation-lab"
      actions={<><AssetPicker assets={assetList} value={selectedAsset} onChange={setSelectedAsset} className="w-48" /><button onClick={handleRun} disabled={running || !selectedAsset} className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded font-medium text-sm hover:bg-primary/90 transition disabled:opacity-50" data-testid="run-validation-btn"><Play className="w-4 h-4" />{running ? 'RUNNING...' : 'RUN VALIDATION'}</button></>}>
      <Tabs defaultValue="tuning" className="w-full">
        <TabsList className="glass-effect mb-6 w-full justify-start" data-testid="validation-tabs"><TabsTrigger value="tuning" className="font-bold text-xs">TUNING RESULTS</TabsTrigger><TabsTrigger value="walkforward" className="font-bold text-xs">WALK-FORWARD</TabsTrigger><TabsTrigger value="hawkes" className="font-bold text-xs">HAWKES STRESS</TabsTrigger><TabsTrigger value="execrisk" className="font-bold text-xs">EXEC RISK</TabsTrigger></TabsList>
        <TabsContent value="tuning"><TuningTab data={tuningData} /></TabsContent>
        <TabsContent value="walkforward"><WalkForwardTab data={walkforwardData} /></TabsContent>
        <TabsContent value="hawkes"><HawkesStressTab data={hawkesData} /></TabsContent>
        <TabsContent value="execrisk"><ExecRiskTab data={execRiskData} /></TabsContent>
      </Tabs>
    </PageShell>
  );
};

export default ValidationLab;
