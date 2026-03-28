import { useState, useEffect, useCallback } from "react";
import {
  Wallet,
  TrendingUp,
  TrendingDown,
  ArrowUpDown,
  Trash2,
  RefreshCw,
  Send,
  DollarSign,
  BarChart3,
  Clock,
  AlertTriangle,
} from "lucide-react";
import { toast } from "sonner";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { PageShell } from "@/components/shared";
import api from "@/api";

const fmt = (n, decimals = 2) =>
  n != null ? `$${Number(n).toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}` : "—";

const pnlColor = (v) =>
  v > 0 ? "text-emerald-400" : v < 0 ? "text-red-400" : "text-muted-foreground";

const pnlBg = (v) =>
  v > 0
    ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
    : v < 0
      ? "bg-red-500/10 text-red-400 border-red-500/20"
      : "bg-muted/50 text-muted-foreground border-border";

function SummaryMetric({ icon: Icon, label, value, sub, accent }) {
  return (
    <div className="flex items-start gap-3">
      <div className="rounded-lg bg-primary/10 p-2.5 mt-0.5">
        <Icon className="h-4 w-4 text-primary" />
      </div>
      <div className="min-w-0">
        <p className="text-xs text-muted-foreground uppercase tracking-wider">{label}</p>
        <p className={`text-xl font-bold font-mono mt-0.5 ${accent || "text-foreground"}`}>
          {value}
        </p>
        {sub && <p className="text-xs text-muted-foreground mt-0.5">{sub}</p>}
      </div>
    </div>
  );
}

function OrderForm({ onSubmit, submitting }) {
  const [symbol, setSymbol] = useState("");
  const [side, setSide] = useState("buy");
  const [quantity, setQuantity] = useState("");
  const [price, setPrice] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!symbol.trim()) return toast.error("Symbol is required");
    const qty = parseFloat(quantity);
    if (!qty || qty <= 0) return toast.error("Enter a valid quantity");
    const limitPrice = price ? parseFloat(price) : undefined;
    if (price && (!limitPrice || limitPrice <= 0)) return toast.error("Enter a valid price");

    onSubmit({
      symbol: symbol.trim().toUpperCase(),
      side,
      quantity: qty,
      price: limitPrice,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="symbol" className="text-xs uppercase tracking-wider text-muted-foreground">
          Symbol
        </Label>
        <Input
          id="symbol"
          placeholder="AAPL, BTC-USD, RELIANCE.NS"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          className="bg-background/50 font-mono"
        />
      </div>

      <div className="space-y-2">
        <Label className="text-xs uppercase tracking-wider text-muted-foreground">Side</Label>
        <div className="grid grid-cols-2 gap-2">
          <Button
            type="button"
            variant={side === "buy" ? "default" : "outline"}
            size="sm"
            className={side === "buy" ? "bg-emerald-600 hover:bg-emerald-700 text-white" : ""}
            onClick={() => setSide("buy")}
          >
            <TrendingUp className="h-3.5 w-3.5 mr-1.5" />
            Buy
          </Button>
          <Button
            type="button"
            variant={side === "sell" ? "default" : "outline"}
            size="sm"
            className={side === "sell" ? "bg-red-600 hover:bg-red-700 text-white" : ""}
            onClick={() => setSide("sell")}
          >
            <TrendingDown className="h-3.5 w-3.5 mr-1.5" />
            Sell
          </Button>
        </div>
      </div>

      <div className="space-y-2">
        <Label htmlFor="quantity" className="text-xs uppercase tracking-wider text-muted-foreground">
          Quantity
        </Label>
        <Input
          id="quantity"
          type="number"
          step="any"
          min="0"
          placeholder="0.00"
          value={quantity}
          onChange={(e) => setQuantity(e.target.value)}
          className="bg-background/50 font-mono"
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="price" className="text-xs uppercase tracking-wider text-muted-foreground">
          Limit Price <span className="normal-case text-muted-foreground/60">(optional)</span>
        </Label>
        <Input
          id="price"
          type="number"
          step="any"
          min="0"
          placeholder="Market order"
          value={price}
          onChange={(e) => setPrice(e.target.value)}
          className="bg-background/50 font-mono"
        />
      </div>

      <Button type="submit" className="w-full" disabled={submitting}>
        {submitting ? (
          <RefreshCw className="h-4 w-4 animate-spin mr-2" />
        ) : (
          <Send className="h-4 w-4 mr-2" />
        )}
        {submitting ? "Placing..." : "Place Order"}
      </Button>
    </form>
  );
}

function PositionsTable({ positions }) {
  if (!positions?.length) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
        <BarChart3 className="h-10 w-10 mb-3 opacity-40" />
        <p className="text-sm">No open positions</p>
        <p className="text-xs mt-1">Place an order to get started</p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-muted-foreground">
            <th className="text-left py-3 px-2 text-xs uppercase tracking-wider">Symbol</th>
            <th className="text-right py-3 px-2 text-xs uppercase tracking-wider">Qty</th>
            <th className="text-right py-3 px-2 text-xs uppercase tracking-wider">Avg Price</th>
            <th className="text-right py-3 px-2 text-xs uppercase tracking-wider">Mkt Value</th>
            <th className="text-right py-3 px-2 text-xs uppercase tracking-wider">Unrealized P&L</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((pos) => {
            const pnl = pos.unrealized_pnl ?? pos.pnl ?? 0;
            return (
              <tr key={pos.symbol} className="border-b border-border/40 hover:bg-white/[0.02] transition-colors">
                <td className="py-3 px-2">
                  <span className="font-mono font-semibold">{pos.symbol}</span>
                </td>
                <td className="py-3 px-2 text-right font-mono">{pos.quantity}</td>
                <td className="py-3 px-2 text-right font-mono">{fmt(pos.avg_price)}</td>
                <td className="py-3 px-2 text-right font-mono">{fmt(pos.market_value ?? pos.current_value)}</td>
                <td className="py-3 px-2 text-right">
                  <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono font-medium border ${pnlBg(pnl)}`}>
                    {pnl > 0 ? "+" : ""}
                    {fmt(pnl)}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function TradeHistoryTable({ trades }) {
  if (!trades?.length) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
        <Clock className="h-10 w-10 mb-3 opacity-40" />
        <p className="text-sm">No trade history</p>
        <p className="text-xs mt-1">Completed trades will appear here</p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-muted-foreground">
            <th className="text-left py-3 px-2 text-xs uppercase tracking-wider">Time</th>
            <th className="text-left py-3 px-2 text-xs uppercase tracking-wider">Symbol</th>
            <th className="text-left py-3 px-2 text-xs uppercase tracking-wider">Side</th>
            <th className="text-right py-3 px-2 text-xs uppercase tracking-wider">Qty</th>
            <th className="text-right py-3 px-2 text-xs uppercase tracking-wider">Price</th>
            <th className="text-right py-3 px-2 text-xs uppercase tracking-wider">Total</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t, i) => {
            const total = (t.quantity ?? t.size ?? 0) * (t.price ?? 0);
            const ts = t.timestamp || t.time || t.created_at;
            return (
              <tr key={`${ts}-${i}`} className="border-b border-border/40 hover:bg-white/[0.02] transition-colors">
                <td className="py-3 px-2 font-mono text-xs text-muted-foreground">
                  {ts ? new Date(ts).toLocaleString() : "—"}
                </td>
                <td className="py-3 px-2 font-mono font-semibold">{t.symbol || "—"}</td>
                <td className="py-3 px-2">
                  <Badge
                    variant="outline"
                    className={
                      t.side === "buy"
                        ? "border-emerald-500/40 text-emerald-400 bg-emerald-500/10"
                        : "border-red-500/40 text-red-400 bg-red-500/10"
                    }
                  >
                    {t.side?.toUpperCase()}
                  </Badge>
                </td>
                <td className="py-3 px-2 text-right font-mono">{t.quantity ?? t.size ?? "—"}</td>
                <td className="py-3 px-2 text-right font-mono">{fmt(t.price)}</td>
                <td className="py-3 px-2 text-right font-mono">{fmt(total)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function PaperTrading() {
  const [portfolio, setPortfolio] = useState(null);
  const [positions, setPositions] = useState([]);
  const [trades, setTrades] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const fetchAll = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    else setRefreshing(true);
    try {
      const [pRes, posRes, tRes] = await Promise.allSettled([
        api.getPaperPortfolio(),
        api.getPaperPositions(),
        api.getPaperTrades(50),
      ]);
      if (pRes.status === "fulfilled") setPortfolio(pRes.value.data);
      if (posRes.status === "fulfilled") {
        const raw = posRes.value.data;
        setPositions(Array.isArray(raw) ? raw : raw?.positions || []);
      }
      if (tRes.status === "fulfilled") {
        const raw = tRes.value.data;
        setTrades(Array.isArray(raw) ? raw : raw?.trades || []);
      }
    } catch {
      toast.error("Failed to load paper trading data");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  const handleOrder = async (order) => {
    setSubmitting(true);
    try {
      await api.placePaperOrder(order);
      toast.success(`${order.side.toUpperCase()} ${order.quantity} ${order.symbol} submitted`);
      fetchAll(true);
    } catch (err) {
      const msg = err?.response?.data?.detail || "Order failed";
      toast.error(msg);
    } finally {
      setSubmitting(false);
    }
  };

  const handleReset = async () => {
    try {
      await api.resetPaperTrading();
      toast.success("Paper trading portfolio reset");
      fetchAll();
    } catch {
      toast.error("Reset failed");
    }
  };

  const cash = portfolio?.cash ?? portfolio?.cash_balance ?? 0;
  const totalValue = portfolio?.total_value ?? portfolio?.portfolio_value ?? cash;
  const pnl = portfolio?.pnl ?? portfolio?.total_pnl ?? 0;
  const posCount = positions.length;

  return (
    <PageShell
      title="Paper Trading"
      subtitle="Simulated portfolio with real-time order execution"
      loading={loading}
      testId="paper-trading"
      actions={
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => fetchAll(true)}
            disabled={refreshing}
          >
            <RefreshCw className={`h-4 w-4 mr-1.5 ${refreshing ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="outline" size="sm" className="text-red-400 border-red-500/30 hover:bg-red-500/10">
                <Trash2 className="h-4 w-4 mr-1.5" />
                Reset
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent className="glass-effect border-border">
              <AlertDialogHeader>
                <AlertDialogTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5 text-red-400" />
                  Reset Paper Trading
                </AlertDialogTitle>
                <AlertDialogDescription>
                  This will clear all positions, trade history, and reset your cash balance
                  to the starting amount. This action cannot be undone.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  onClick={handleReset}
                  className="bg-red-600 hover:bg-red-700 text-white"
                >
                  Reset Everything
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      }
    >
      {/* ── Portfolio Summary ────────────────────────────── */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 mb-6">
        <Card className="glass-effect">
          <CardContent className="pt-5 pb-4">
            <SummaryMetric icon={Wallet} label="Cash Balance" value={fmt(cash)} />
          </CardContent>
        </Card>
        <Card className="glass-effect">
          <CardContent className="pt-5 pb-4">
            <SummaryMetric icon={DollarSign} label="Portfolio Value" value={fmt(totalValue)} />
          </CardContent>
        </Card>
        <Card className="glass-effect">
          <CardContent className="pt-5 pb-4">
            <SummaryMetric
              icon={pnl >= 0 ? TrendingUp : TrendingDown}
              label="Total P&L"
              value={`${pnl >= 0 ? "+" : ""}${fmt(pnl)}`}
              accent={pnlColor(pnl)}
            />
          </CardContent>
        </Card>
        <Card className="glass-effect">
          <CardContent className="pt-5 pb-4">
            <SummaryMetric
              icon={ArrowUpDown}
              label="Open Positions"
              value={posCount}
              sub={`${trades.length} total trades`}
            />
          </CardContent>
        </Card>
      </div>

      {/* ── Main Content Grid ───────────────────────────── */}
      <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
        {/* Order Form */}
        <Card className="glass-effect h-fit">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <Send className="h-4 w-4 text-primary" />
              PLACE ORDER
            </CardTitle>
          </CardHeader>
          <CardContent>
            <OrderForm onSubmit={handleOrder} submitting={submitting} />
          </CardContent>
        </Card>

        {/* Right Column — Positions + History */}
        <div className="space-y-6">
          <Card className="glass-effect">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-primary" />
                OPEN POSITIONS
                {posCount > 0 && (
                  <Badge variant="secondary" className="ml-auto text-xs">
                    {posCount}
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <PositionsTable positions={positions} />
            </CardContent>
          </Card>

          <Card className="glass-effect">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Clock className="h-4 w-4 text-primary" />
                TRADE HISTORY
                {trades.length > 0 && (
                  <Badge variant="secondary" className="ml-auto text-xs">
                    {trades.length}
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <TradeHistoryTable trades={trades} />
            </CardContent>
          </Card>
        </div>
      </div>
    </PageShell>
  );
}

export default PaperTrading;
