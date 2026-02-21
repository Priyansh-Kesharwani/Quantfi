# QuantFi Portfolio Simulation Engine — Deep Research Prompt

## Context & Background

QuantFi is pivoting from a DCA (Dollar-Cost Averaging) intelligence platform to a **portfolio simulation and backtesting engine** with dynamic entry/exit signals across a multi-asset universe.

### What Already Exists (Strengths to Preserve)

- **Composite Scoring Engine** (0-100): Combines 4 sub-scores via weighted sum:
  - Technical Momentum (40%): SMA crossovers, RSI oversold/overbought, MACD divergence, Bollinger Band positioning, ADX trend strength
  - Volatility Opportunity (20%): ATR percentile, drawdown depth
  - Statistical Deviation (20%): Multi-horizon z-scores (20/50/100-day)
  - Macro FX (20%): USD-INR rate deviation from historical mean
- **Real Market Data Pipeline**: Yahoo Finance (yfinance) fetching historical OHLCV for equities, ETFs, commodities, crypto, indices — both US and Indian markets
- **Multi-Asset Watchlist**: Users manage a universe of assets (AAPL, SPY, GC=F, RELIANCE.NS, TCS.NS, NFLX, SI=F, ^NSEI)
- **Sentiment Agent**: LLM-powered sentiment analysis with 6-layer asset-relevance filtering
- **Existing Backtester Modules**:
  - `DCAPortfolioSimulator`: Score-tiered DCA with transaction costs, slippage, equity curves, uniform-DCA comparison
  - `DiagnosticBacktester`: Decile analysis (score vs forward returns), crisis regime analysis, drawdown analysis
  - `PurgedKFold` + `WalkForwardCV`: Time-series cross-validation with embargo zones
  - `BlockBootstrap`: Confidence intervals for backtest statistics

### What's Missing (The 4 Critical Gaps)

**Gap 1 — Exit Signal Framework**: The scoring engine is purely a buy-opportunity detector (rewards: below SMA, oversold RSI, deep drawdowns, negative z-scores). There is NO exit/sell-side intelligence. A score of 20 means "not a good time to enter" — NOT "you should exit." The system needs a dedicated exit mechanism that handles both risk management (stop losses) and profit-taking.

**Gap 2 — Position State Machine**: The architecture is entirely stateless. Each API call computes independently. There is no concept of "currently holding 50 shares of AAPL entered at $175" or portfolio state tracking. A day-by-day simulation engine needs: cash tracking, position management, trade execution with costs, and equity curve generation at the portfolio level.

**Gap 3 — Mean-Reversion Strategy Limitation**: The scoring engine exclusively detects depressed-price conditions. During sustained uptrends (SPY 2013-2019, NVDA 2023-2024), the score stays low (30-45) because there's no dip to buy. The system will sit in cash during the strongest bull runs. This is inherently a mean-reversion strategy, which is valid but must be acknowledged and properly benchmarked.

**Gap 4 — Transaction Cost Calibration**: The platform serves retail investors (Indian market, ₹5,000-50,000 position sizes). The institutional √volume market impact model is irrelevant. Need: retail-calibrated costs (brokerage, bid-ask spread, forex conversion, short-term capital gains tax implications).

### Tech Stack Constraints
- Backend: Python 3.11+, FastAPI, MongoDB (Motor), NumPy, Pandas
- Frontend: React 18, Tailwind CSS, Recharts, Lucide icons
- Data: yfinance for historical OHLCV, no tick-level data, daily resolution
- Scoring runs per-asset, not cross-asset (no correlation-aware scoring yet)
- No ML models in the scoring engine — purely rule-based indicator aggregation

---

## Research Areas

For each area below, I need:
1. **Academic/industry foundations** — What does the quantitative finance literature say? What do practitioners actually use?
2. **Concrete implementation design** — Data structures, algorithms, parameter choices with justification
3. **Failure modes & edge cases** — What breaks, when does it underperform, what are the known pitfalls?
4. **Value proposition** — How does this make the platform more compelling vs. generic backtesting tools (Backtrader, VectorBT, QuantConnect)?

---

## RESEARCH AREA 1: Exit Signal Architecture for a Mean-Reversion Entry System

### The Problem
Given that entry signals detect "price is depressed" (high score → buy), what is the optimal exit framework? The exit must serve two distinct jobs:
- **Risk management**: Prevent catastrophic losses when mean-reversion fails (the dip keeps dipping)
- **Profit-taking**: Capture gains when the reversion has played out (price recovered to mean)

### Questions to Research

#### 1.1 — Trailing Stop Mechanics for Mean-Reversion Strategies
- What is the optimal trailing stop percentage for mean-reversion strategies vs trend-following? Literature suggests mean-reversion needs tighter stops (5-10%) vs trend-following (15-25%). Validate this.
- Should the trailing stop be **fixed percentage** (e.g., 8% from peak), **ATR-based** (e.g., 2× ATR from peak), or **volatility-adjusted** (e.g., 1.5σ)? Which one adapts best to different asset classes (equities vs commodities vs crypto)?
- **Chandelier Exit** (highest high − N × ATR): Is this appropriate for our use case? How does it compare to fixed-percentage trailing stops in mean-reversion backtests?
- How should the trailing stop activate? Immediately on entry? After a minimum gain threshold (e.g., start trailing only after +3% unrealized gain)? Research "breakeven stop" vs "immediate trail."

#### 1.2 — Score-Based Take-Profit as Mean-Reversion Completion Signal
- The entry score naturally declines as the asset recovers from its dip (drawdown shrinks, RSI normalizes, z-scores approach zero). Can we formalize this as: "exit when score drops below X% of entry score"?
- What is the theoretical basis for using the same mean-reversion indicator set for both entry AND exit timing? Research Ornstein-Uhlenbeck mean-reversion models and how they define "reversion complete."
- Should the take-profit threshold be **absolute** (exit when score < 35) or **relative to entry** (exit when score < entry_score × 0.5)?
- How does this interact with the trailing stop? Which fires first in practice? Research priority/hierarchy of exit conditions.

#### 1.3 — Time-Based Exit Decay
- Mean-reversion trades have a natural "shelf life" — if the reversion hasn't happened in N days, the thesis is broken. What does the literature say about optimal holding periods for mean-reversion strategies?
- Research "time stop" or "time decay exit": after N days, begin reducing position or force exit regardless of score/price. What N works for daily-resolution data across equities, commodities, and indices?
- How do institutional mean-reversion desks handle aging positions?

#### 1.4 — Composite Exit Design
- How should trailing stop + score-threshold + time-decay be combined? Research multi-rule exit systems:
  - **OR logic**: exit if ANY condition fires (most conservative)
  - **AND logic with priority**: trailing stop always fires; score threshold AND time decay must both agree (more permissive)
  - **Tiered exits**: partial position reduction at different thresholds
- What does the academic literature say about the marginal value of adding exit rules? Diminishing returns? Overfitting risk?

#### 1.5 — Failure Modes
- **Whipsaw**: Price dips → entry → partial recovery → trailing stop fires → price continues up. How frequent is this for mean-reversion? What's the false-exit rate?
- **Gap risk**: Asset gaps down past trailing stop (common in earnings, commodities). How to handle? Research "gap-aware stops" and whether daily-resolution data even captures this.
- **Regime dependence**: Exit rules that work in 2015-2020 may fail in 2020-2025. Research regime-conditional exit tuning.

---

## RESEARCH AREA 2: Portfolio-Level Position State Machine Architecture

### The Problem
Transform a collection of per-asset scores into a coherent multi-asset portfolio with proper cash management, position sizing, and state tracking.

### Questions to Research

#### 2.1 — State Machine Design for Event-Driven Portfolio Simulation
- What is the minimal state representation needed for a daily-resolution portfolio simulator?
  ```
  State = {
    date, cash, 
    positions: {symbol → {units, entry_price, entry_date, entry_score, peak_price_since_entry}},
    pending_signals: [...],
    trade_log: [...],
    daily_snapshots: [...]
  }
  ```
- Research how VectorBT, Zipline, and Backtrader represent portfolio state. What can we simplify given we only need daily resolution (no intraday)?
- How should the state machine handle **simultaneous signals** (e.g., 3 assets all trigger entry on the same day but insufficient cash for all 3)?
- Research **priority queue** for signal processing: should we enter the highest-score asset first? Or the most depressed? Or random?

#### 2.2 — Position Sizing for Multi-Asset Mean-Reversion
- **Equal-weight**: Each new entry gets `cash / max_positions` allocation. Simple but ignores conviction.
- **Score-weighted**: Higher entry score → larger allocation. `size_i = (score_i / Σ scores) × available_cash`. Research whether conviction-weighted sizing improves mean-reversion returns.
- **Volatility-adjusted (risk parity)**: `size_i ∝ 1/σ_i` so each position contributes equal risk. Research whether this improves Sharpe for a mean-reversion portfolio vs equal-weight.
- **Kelly Criterion adaptation**: Can we use the score-to-forward-return relationship (from the existing decile analysis) to estimate edge and apply fractional Kelly? Is this robust for rule-based (not ML) scoring?
- What is the optimal **max positions** count for a retail portfolio? Research concentration vs diversification tradeoffs for 5-20 asset universes.

#### 2.3 — Cash Management & Opportunity Cost
- In mean-reversion, the portfolio may be 80%+ cash during calm markets (no dips to buy). How should we handle cash drag?
- Options: (a) Ignore it (cash earns 0), (b) Apply a risk-free rate to cash (India: ~6.5%, US: ~5%), (c) Invest idle cash in a benchmark ETF
- Research how professional mean-reversion funds handle cash buffers. What is the typical cash allocation range?

#### 2.4 — Rebalancing & Signal Processing Frequency
- Should signals be checked daily, weekly, or monthly? Research the tradeoff between signal responsiveness and transaction cost drag.
- For daily-resolution data, is there an optimal "minimum holding period" to prevent overtrading? Research minimum rebalance intervals for mean-reversion.
- How to handle rebalance day when multiple exits and entries coincide? Research "netting" (settle all exits first, then enter with proceeds) vs "simultaneous" (all at once, may need margin).

#### 2.5 — Portfolio-Level Risk Controls
- **Max drawdown circuit breaker**: If portfolio drawdown exceeds X%, halt all new entries and liquidate? Research whether this improves or hurts long-term mean-reversion returns (it may prevent buying the deepest dips).
- **Correlation-aware position limits**: If AAPL and NFLX are 85% correlated, should we limit combined exposure? Research correlation-based position limits for small portfolios.
- **Sector/asset-class concentration limits**: Max 40% in equities, 20% in commodities, etc. Research whether this improves risk-adjusted returns in multi-asset mean-reversion.

---

## RESEARCH AREA 3: Making Mean-Reversion Strategy Honest & Benchmarkable

### The Problem
The scoring engine will only enter during dips. During sustained uptrends, it sits in cash. This needs to be honestly benchmarked and clearly communicated — it's the #1 source of user confusion ("why did the simulator underperform SPY?").

### Questions to Research

#### 3.1 — Proper Benchmarks for Mean-Reversion Strategies
- **Buy-and-hold** (equal-weight across universe at start): The naive benchmark. Mean-reversion will underperform during bull runs. Research how to present this honestly.
- **60/40 portfolio** (60% equities, 40% bonds): More realistic benchmark for a strategy that holds significant cash. Research whether mean-reversion's cash drag makes it comparable to a conservative allocation.
- **Risk-adjusted benchmarks**: Sharpe ratio, Sortino ratio, Calmar ratio (CAGR/maxDD). Research which risk-adjusted metric best showcases mean-reversion's advantage (typically: lower drawdowns, better risk-adjusted returns even if absolute returns lag).
- **Same-universe uniform timing**: Buy every asset at regular intervals regardless of score (the uniform DCA benchmark already exists in the codebase). This directly measures the VALUE-ADD of the scoring engine.

#### 3.2 — Regime-Conditional Performance Reporting
- Research how to partition backtest results into regimes: **bull** (market up >15%/yr), **bear** (market down >15%/yr), **sideways** (±15%), **high-vol** (VIX > 25), **low-vol** (VIX < 15).
- Mean-reversion should outperform in high-vol and sideways regimes, underperform in low-vol bull regimes. Quantify this expected pattern.
- Research "regime-conditional Sharpe ratio" — is this a standard metric?
- How do professional quantitative strategy reports present regime-dependent performance? Research Bridgewater, AQR, and Man Group's strategy reporting format.

#### 3.3 — Alpha Decomposition
- Given the portfolio's return stream, decompose into:
  - **Market beta** (exposure to broad market moves)
  - **Timing alpha** (value from entering during dips vs random)
  - **Selection alpha** (value from choosing which assets to overweight)
  - **Cost drag** (how much returns are eroded by transaction costs)
- Research Brinson-Fachler attribution or similar frameworks adapted for active timing strategies (not traditional asset allocation).
- What's the minimum number of trades/signals needed for alpha decomposition to be statistically meaningful?

#### 3.4 — Statistical Significance of Backtest Results
- Research how to determine if the simulation's outperformance is statistically significant vs luck:
  - **Bootstrap hypothesis test**: Resample returns, calculate p-value of observed alpha
  - **Permutation test**: Randomly shuffle entry signals, compare to actual signal performance
  - **Deflated Sharpe Ratio** (Bailey & López de Prado): Adjusts for multiple testing, overfitting
- What minimum backtest length (years) and minimum number of trades gives reliable results for a multi-asset mean-reversion strategy?
- Research the "minimum backtest length" literature — Marcos López de Prado's work on backtest overfitting.

#### 3.5 — Clearly Communicating "The Strategy Sat in Cash for 3 Years"
- How should the UI present periods of inactivity? Research how robo-advisors and quant platforms communicate "the model is waiting for opportunity."
- Research UX patterns for: cash allocation timeline, "opportunity drought" periods, expected time-in-market ratios for mean-reversion.
- What is the expected time-in-market for a mean-reversion strategy on US equities? On commodities? On a mixed universe? (Literature suggests 20-40% of the time for equities.)

---

## RESEARCH AREA 4: Retail-Calibrated Transaction Cost Model

### The Problem
Transform the institutional cost model (√volume impact, queue-position slippage) into one that reflects actual costs for Indian retail investors trading $50-$5,000 per position through platforms like Groww, Zerodha, Vested, INDmoney.

### Questions to Research

#### 4.1 — Actual Cost Structure for Indian Retail Cross-Border Investing
- Research the EXACT fee structure for:
  - **Zerodha** (Indian equities): ₹20/trade flat or 0.03%, STT 0.1% (sell), stamp duty, SEBI turnover fee
  - **Vested Finance** (US equities from India): Forex spread (~0.5-1%), wire transfer fees, no per-trade commission?
  - **INDmoney** (US equities from India): Forex markup, any platform fees?
  - **Groww** (US stocks via Stockal): Commission structure, forex conversion cost
- What is the **total round-trip cost** (buy + sell) for a ₹10,000 position in AAPL traded from India?
- What is the **total round-trip cost** for a ₹10,000 position in RELIANCE.NS on NSE?

#### 4.2 — Simplified Cost Model Design
- Research whether a single "bps_per_round_trip" parameter can adequately capture retail costs:
  - US equities from India: ~100-200 bps round-trip (forex spread dominates)
  - Indian equities: ~30-50 bps round-trip (STT + brokerage)
  - Commodities (futures): ~20-40 bps
  - Crypto: ~50-150 bps (exchange spread + network fees)
- Should the model be **per-asset-class** or **per-trade-size**? Research whether cost varies meaningfully with position size in the $50-$5,000 range (likely not for market impact, yes for fixed fees).

#### 4.3 — Tax Impact on Active Trading Returns
- **Indian tax on US stocks**: Short-term (<24 months) gains taxed as income slab (up to 30%). Long-term: 20% with indexation. Research: how does this affect the simulation's realistic returns?
- **Indian tax on Indian stocks**: Short-term (<12 months) at 15%, long-term at 10% above ₹1L exemption.
- Should the simulator show **pre-tax** and **post-tax** returns? Research how competing platforms (Smallcase, Kuvera) handle this.
- Research: for a mean-reversion strategy with average holding period of 20-60 days, nearly ALL gains will be short-term. Quantify the tax drag.

#### 4.4 — Slippage Model for Daily-Resolution Simulation
- Since we use daily OHLCV (no intraday), we can't model exact execution price. Research standard approaches:
  - **Close-price execution**: Assume trade at daily close (unrealistic — you didn't know the close when you decided)
  - **Next-day open**: Signal on day T close, execute at day T+1 open (most realistic for daily signals)
  - **VWAP proxy**: Approximate with `(high + low + close) / 3` (common in backtesting literature)
- Research which execution assumption most closely matches reality for retail investors who set market orders in the morning based on previous close.
- What systematic bias does each assumption introduce?

---

## RESEARCH AREA 5: Differentiated Value Proposition vs. Existing Backtesting Tools

### The Problem
Why would someone use QuantFi's portfolio simulator instead of Backtrader, VectorBT, QuantConnect, or TradingView's Pine Script? We need a clear value proposition that justifies the platform's existence after the pivot.

### Questions to Research

#### 5.1 — Competitive Landscape Analysis
- Research the strengths and weaknesses of:
  - **VectorBT**: Fast vectorized backtesting, Python, good for parameter sweeps. Weakness?
  - **Backtrader**: Event-driven, flexible, Python. Weakness?
  - **QuantConnect/Lean**: Cloud-based, multi-asset, live trading. Weakness?
  - **Smallcase** (India-specific): Thematic portfolio builder. How does it handle backtesting?
  - **Streak by Zerodha**: Visual algo builder. What does it offer for backtesting?
  - **TradingView**: Pine Script backtesting. Limitations?
- What gap exists that none of these fill? Research user complaints and feature requests for each.

#### 5.2 — Potential Differentiators for QuantFi
- **Sentiment-augmented scoring**: Your existing sentiment agent (LLM-powered, 6-layer filtering) is genuinely novel. Research: do any competing platforms combine rule-based technical scoring with LLM sentiment analysis for entry timing? This could be the moat.
- **Indian investor focus**: Cross-border cost modeling (INR→USD), Indian tax implications, NSE/BSE asset support. Research: is there ANY backtesting tool that properly models the Indian retail cross-border investing experience?
- **Score transparency**: Your scoring engine is fully interpretable (rule-based, not black-box ML). Users can see exactly WHY an entry was triggered. Research: how much do retail users value interpretability vs. "just trust the algorithm"?
- **Multi-asset class in one place**: Equities + ETFs + commodities + crypto + indices in a single portfolio simulator. Research: which competing tools support this breadth?

#### 5.3 — What Would Make a User Choose This Over a Spreadsheet?
- The honest question: a motivated user could build a mean-reversion backtest in a Jupyter notebook in 2 hours. Why use QuantFi?
- Research the "build vs. buy" decision for retail quant tools. What features push users from DIY to platform?
- Likely answers to validate: (a) pre-built scoring engine they'd spend weeks building, (b) real-time dashboard (not just historical backtest), (c) sentiment integration, (d) Indian market support, (e) beautiful UI that non-programmers can use.

---

## Deliverables Expected From This Research

For each research area, produce:

1. **Literature summary**: 3-5 key academic papers or practitioner resources with specific findings relevant to our implementation
2. **Recommended approach**: The specific design choice with parameter ranges (e.g., "trailing stop at 2× 14-day ATR, based on [paper/evidence]")
3. **Implementation sketch**: Pseudocode or data structure design that integrates with the existing QuantFi codebase (Python, NumPy/Pandas, FastAPI)
4. **Risk register**: What can go wrong, how likely, and how to detect it in production
5. **Value proposition statement**: One sentence explaining why THIS specific feature makes QuantFi worth using over alternatives

---

## Constraints to Respect

- **Daily resolution only**: No intraday data, no tick-level simulation. All signals and execution at daily OHLCV granularity.
- **Retail position sizes**: $50-$5,000 per trade. No institutional considerations (market impact, block trading, dark pools).
- **Rule-based scoring**: The entry scoring engine is NOT machine learning. It's a weighted sum of hand-crafted indicator rules. Any exit framework should match this philosophy (interpretable, configurable rules — not a black-box neural net).
- **Web-responsive performance**: The simulation must complete in <10 seconds for 5 assets × 5 years of daily data. No hour-long Monte Carlo runs.
- **Indian retail investor persona**: Primary user is investing ₹10K-1L/month across US and Indian markets through discount brokers. Not a professional trader, not an institution.
