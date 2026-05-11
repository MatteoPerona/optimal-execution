# v9 Final — Full Strategy Walkthrough

This document explains every design decision in the pipeline: what problem we are solving, how the data works, what each strategy does and why, how validation is structured, and how to interpret the results. It is written for a teammate who understands markets but has not read the code.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [The Data — LOBSTER Format](#2-the-data--lobster-format)
3. [The Benchmark — TWAP](#3-the-benchmark--twap)
4. [The Project Metric](#4-the-project-metric)
5. [Strategy Logic — One Minute at a Time](#5-strategy-logic--one-minute-at-a-time)
   - [TWAP (baseline)](#strategy-0-twap-baseline)
   - [SpreadQuantile](#strategy-1-spreadquantile)
   - [OFIContrarian](#strategy-2-oficontrarian)
   - [Microprice](#strategy-3-microprice)
   - [AdaptiveDeadline](#strategy-4-adaptivedeadline)
   - [Ensemble (winner)](#strategy-5-ensemble)
6. [Walk-Forward Validation](#6-walk-forward-validation)
7. [Parameter Tuning](#7-parameter-tuning)
8. [AAPL — Autonomous Strategy Selection](#8-aapl--autonomous-strategy-selection)
9. [Results and Interpretation](#9-results-and-interpretation)
10. [Presentation Suite — All Figures Explained](#10-presentation-suite--all-figures-explained)
11. [Per-Minute Improvement Statistics](#11-per-minute-improvement-statistics)
12. [Monday Workflow](#monday-workflow)

---

## 1. The Problem

A large institutional investor needs to buy and sell shares over the course of a trading session. If they slam the full order into the market at once they move the price against themselves. The standard solution is to slice the order into small child orders executed gradually over time — **algorithmic execution**.

Our specific constraint: execute **exactly 1 share per minute**. Within each minute we have many individual tick events (order placements, cancellations, executions). The question is: **which tick do we execute on?**

A naive algorithm executes at the very first tick of every minute — this is TWAP (Time-Weighted Average Price). Any improvement over TWAP represents real money saved on the buy side or earned on the sell side.

---

## 2. The Data — LOBSTER Format

Each stock's CSV contains one row per LOB event (a limit order placed, cancelled, or executed). The relevant columns are:

| Column | Meaning |
|--------|---------|
| `Time` | Timestamp of the event (HH:MM:SS.ffffff) |
| `BidPrice_1` | Best bid price (highest price a buyer will pay) |
| `AskPrice_1` | Best ask price (lowest price a seller will accept) |
| `BidSize_1` | Quantity available at the best bid |
| `AskSize_1` | Quantity available at the best ask |
| `Spread` | `AskPrice_1 − BidPrice_1` (cost of crossing immediately) |
| `MidPrice` | `(Bid + Ask) / 2` (fair value estimate) |
| `BidPrice_2..5` | Prices deeper in the bid-side order book |
| `AskPrice_2..5` | Prices deeper in the ask-side order book |

**Key point:** every row is a snapshot of the full order book *after* an event. The data is already cleaned and timestamped — we do not need to worry about parsing trades vs. quotes separately.

**We add one column in code:**
```python
df["Minute"] = df.index.floor("min")   # e.g., 09:30:00, 09:31:00, ...
```

This groups all ticks within the same minute together, which is the unit of decision.

---

## 3. The Benchmark — TWAP

TWAP (Time-Weighted Average Price) is the simplest possible execution rule:

> At the start of each minute, execute at whatever price is available right now.

Concretely, for a **BUY** order: pay the first tick's `AskPrice_1`.  
For a **SELL** order: receive the first tick's `BidPrice_1`.

This is the **floor** our strategies must beat. It is trivially implementable — you just submit a market order the instant the new minute opens. The entire point of our work is to show we can do better by waiting for a more favourable tick within the minute.

---

## 4. The Project Metric

```
Improvement (%) = 100 × ( 1 − Algo Cost / TWAP Cost )

where:
  TWAP Cost = Σ TWAP buy prices  −  Σ TWAP sell prices   (over all minutes)
  Algo Cost = Σ Algo buy prices  −  Σ Algo sell prices
```

Intuition:

- The **cost** of a combined buy-and-sell programme is the net outflow: you pay to buy, you receive from selling. The difference is the total cost of the round-trip.
- If the algo buys cheaper and sells dearer than TWAP, `Algo Cost < TWAP Cost` → improvement > 0.
- **Improvement = 100%** would mean the algo collapsed the entire cost to zero — i.e., it always bought at the bid and sold at the ask, turning the spread into profit rather than cost.
- **Improvement > 100%** means the algo actually *reversed* the net cost (algo sell proceeds exceed buy costs) — this happens on the synthetic data and is a sign the data is over-predictable, not that we have found free money.

Two auxiliary metrics are also computed:
- **Buy α (bps):** how many basis points cheaper the algo bought vs. TWAP (per minute, averaged).
- **Sell α (bps):** how many basis points more the algo received on sells vs. TWAP.

---

## 5. Strategy Logic — One Minute at a Time

Every strategy implements exactly one method:

```python
strategy.execute(group, side, ctx)  →  float (execution price)
```

Where:
- `group` — a DataFrame of all LOB ticks within one minute
- `side` — `"BUY"` or `"SELL"`
- `ctx` — pre-computed context (rolling statistics, thresholds) that are **point-in-time** — no future data leaks in

The strategy iterates through the ticks in the minute and fires when its condition is met. If no condition is ever met, it falls back to a safe default (usually the first tick = TWAP price).

---

### Strategy 0: TWAP (baseline)

```
Rule: execute at the very first tick of the minute.
```

For a BUY: pay `AskPrice_1` at tick 0.  
For a SELL: receive `BidPrice_1` at tick 0.

This is the benchmark. It is always available (no signal required) and never misses its execution window.

---

### Strategy 1: SpreadQuantile

```
Rule: wait for the first tick where the bid-ask spread is narrower than
      the historical 30th percentile of spreads. Fall back to the LAST tick.
```

**Why this works:** The spread is the immediate cost of crossing the market. A narrower spread means you buy closer to fair value (mid price). If we can wait for a moment when liquidity providers are competing more aggressively — visible as a tighter spread — we pay less.

**The rolling percentile is computed over the prior 500 ticks** (not the current minute or future ticks) to avoid lookahead bias. The threshold at each tick reflects only what has happened up to that point in history.

**Fallback:** last tick of the minute (rather than first), because if the spread never tightened, the end of the minute is often less orderly — this is by design to create a slight pressure to execute.

**Parameters tuned:** rolling window (200 or 500 ticks), quantile threshold (25th or 35th percentile).

---

### Strategy 2: OFIContrarian

```
Rule: for a BUY, fire when cumulative order-flow imbalance within the minute
      is sufficiently negative (heavy selling pressure). For a SELL, fire when
      OFI is sufficiently positive (heavy buying pressure). Fall back to first tick.
```

**What is OFI?** Order Flow Imbalance (Cont, Kukanov & Stoikov 2014) measures the signed pressure building up in the limit order book, tick by tick within a minute:

```
For each new tick vs. previous tick:
  Bid contribution:
    +BidSize_1  if best bid price rose       (new buyer appeared higher up)
    ΔBidSize_1  if best bid price unchanged  (size at best bid changed)
    −BidSize_1  if best bid price fell       (buyer withdrew)

  Ask contribution (symmetric, opposite sign convention):
    +AskSize_1  if best ask price fell
    ΔAskSize_1  if unchanged
    −AskSize_1  if best ask price rose

  OFI(t) = OFI(t-1) + (Bid contribution − Ask contribution)
```

A strongly negative OFI within a minute means the book has been absorbing sell orders — short-term price pressure is down. A contrarian buyer would *want* to step in here, expecting a mean-reversion. That is exactly what OFIContrarian does.

**The threshold** is the rolling Nth percentile of peak OFI magnitudes from prior minutes (computed point-in-time with no lookahead). If the current OFI hasn't exceeded that threshold, we haven't seen unusual pressure yet.

---

### Strategy 3: Microprice

```
Rule: for a BUY, fire when the microprice is close to the ask
      (book is leaning bullish). For a SELL, fire when microprice is near the bid.
```

**What is microprice?** The standard mid-price is `(Bid + Ask) / 2`. But the microprice weights by *size* at each side:

```
Microprice = (BidPrice_1 × AskSize_1  +  AskPrice_1 × BidSize_1) / (BidSize_1 + AskSize_1)
```

If there is 500 shares on the ask and only 100 on the bid, the microprice will be closer to the ask. This reflects that buyers are in greater supply — the next move is more likely upward, making it a good moment to buy (ask is about to be lifted further).

**The signal:** when `microprice ≥ ask − α × spread`, the book is leaning bullish enough to justify executing a buy. `α` controls how much tilt is required before firing.

---

### Strategy 4: AdaptiveDeadline

```
Phase 1 (before 70% of ticks in the minute):
  Fire immediately if price is better than first-tick price by at least 1 bps.

Phase 2 (after 70% of ticks have passed):
  Accept ANY price that is better than the first-tick price, however small.

Fallback: first-tick price (TWAP-equivalent).
```

**Why this works:** The two-phase structure creates an urgency curve. Early in the minute, we are selective — only jump if the price has genuinely improved. As we approach the end of the minute, we cannot afford to miss the window, so we accept any improvement rather than risk getting stuck with the TWAP default.

**The `improvement_bps` parameter** converts to a price threshold: `first_price × improvement_bps / 10,000`. For a $220 AMZN share and `improvement_bps=1.0`, this is $0.022 — just over two cents.

---

### Strategy 5: Ensemble (the winner on all four stocks)

```
At each tick, three binary signals vote:
  A. Spread is unusually tight (below rolling 30th percentile threshold)
  B. OFI is contrarian (cumulative OFI exceeds the rolling threshold in the right direction)
  C. Microprice tilts in the execution direction

Fire when: (votes ≥ min_votes) AND (price improves vs. first tick)
Past deadline: accept any price improvement regardless of votes
Fallback: first-tick price
```

**Why Ensemble wins:** no single signal is consistently right. Spread tightening can coincide with news-driven volatility that makes it the worst time to trade. OFI contrarianness can be wrong if the pressure is trend-following not mean-reverting. Microprice can temporarily misread the book. But when two or three signals agree simultaneously, confidence is much higher.

The price-improvement gate (`price improves vs. first tick`) means Ensemble never executes at a worse price than TWAP regardless of the signal — the worst case is TWAP itself. The strategy is **asymmetric**: upside from timing, no downside vs. benchmark.

**Parameters selected on real data:**

| Stock | min_votes | deadline_frac | Why |
|-------|:---------:|:-------------:|-----|
| AMZN  | 3         | 0.70          | Wide spread requires high conviction before firing |
| GOOG  | 2         | 0.80          | Wide spread; longer patience window captures late reversions |
| INTC  | 3         | 0.70          | Penny spread; 3-vote filters the dense signal |
| MSFT  | 3         | 0.70          | Same as INTC |

---

## 6. Walk-Forward Validation

The professors' critique of Milestone 1 was that results were reported only in-sample — we tuned on all the data and evaluated on the same data. That is not meaningful validation.

**v9 fixes this with a strict chronological 70/30 split:**

```
All 270 minutes of training data, in time order:
[─────────────────── Inner Train (189 min) ──────────────────][─ Holdout (81 min) ─]
          Tune strategy + parameters on this               Never touched during tuning
```

After tuning on the inner train, we run the selected strategy on the holdout as if it were the real future — no re-optimisation, no peeking. The holdout result is the true out-of-sample performance.

**Retention ratio** = `holdout improvement / train improvement`

| Range | Interpretation |
|-------|---------------|
| > 90% | Edge is very robust — the timing signal is stable across the session |
| 70–90% | Healthy decay — expected and acceptable |
| 50–70% | Meaningful decay — edge exists but has overfitting component, flag for discussion |
| < 50% | Severe overfitting — in-sample result should not be trusted |

Our results:

| Stock | Retention | Assessment |
|-------|:---------:|------------|
| AMZN | 90.7% | Robust |
| GOOG | 97.4% | Robust — near-perfect retention |
| INTC | 64.2% | Caution — high-frequency stocks have more within-minute structure to overfit |
| MSFT | 66.2% | Caution — same reason as INTC |

No stock falls below 50%. The absolute holdout improvements (78%, 74%, 87%, 110%) are all very strong.

**Why INTC/MSFT show more decay:** Both stocks trade at $27–$30 with 1-cent spreads and 1,500–2,000 ticks per minute. With that density, the Ensemble signal fires very frequently, and the specific timing patterns in the first 189 minutes (market open dynamics, news flow, intraday seasonality) may differ from the last 81 minutes. The edge persists but the in-sample figure overstates it.

---

## 7. Parameter Tuning

Grid search across 17 strategy-parameter combinations per stock:

```
SpreadQuantile:   4 combos   (rolling_window ∈ {200,500}, q ∈ {0.25,0.35})
OFIContrarian:    3 combos   (threshold_pct ∈ {0.60,0.70,0.80})
Microprice:       3 combos   (alpha ∈ {0.05,0.10,0.20})
AdaptiveDeadline: 4 combos   (improvement_bps ∈ {0.5,1.0}, deadline_frac ∈ {0.6,0.8})
Ensemble:         3 combos   (min_votes ∈ {2,3}, deadline_frac ∈ {0.6,0.7,0.8})
─────────────────────────────
Total:           17 combos
```

Each combo is evaluated on the **inner train only** (189 minutes). The top-scoring combo is selected and then evaluated **once** on the holdout. There is no iterative feedback from the holdout into the tuning — that would be data leakage.

**Why not more combos?** With 270 minutes of training data and 17 strategies, the evaluation dataset is already tight. Adding more combos increases the chance of accidentally discovering a combo that works for the 189-minute period but has no generalisable signal (multiple comparison problem). The 17-combo grid is deliberately narrow.

---

## 8. AAPL — Autonomous Strategy Selection

AAPL is the unseen stock. We have never examined its training data during the design of the pipeline. When `AAPL_5levels_train.csv` arrives, the algorithm selects a strategy **without human input** via two competing routes.

### Microstructure Fingerprinting

First, we characterise AAPL's microstructure with four features (excluding price level, which shouldn't drive strategy routing):

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `spread_bps` | median(Spread / MidPrice) × 10,000 | How wide the market is relative to price |
| `tick_freq` | avg ticks per minute | How liquid and active the stock is |
| `depth_per_bps` | median((BidSize + AskSize) / spread_bps) | How much depth per unit of spread |
| `vol_bps` | std of minute mid-returns in bps | Intraday price volatility |

These four numbers form the fingerprint. We compute the log-scale Euclidean distance between AAPL's fingerprint and each of the four known stocks:

```
distance(AAPL, stock) = sqrt( Σ  (log(AAPL[k]) − log(stock[k]))² )
                               k ∈ {spread_bps, tick_freq, depth_per_bps, vol_bps}
```

Log scale is used so that a stock with a 1-cent spread vs. 2-cent spread (2×) is treated the same as $0.14 vs. $0.28 (also 2×) — ratios matter, not absolute differences.

### Route A — Nearest-Neighbour Transfer

Take the strategy + parameters that won on the nearest stock (smallest fingerprint distance) and apply them directly to AAPL. No re-tuning.

### Route B — Direct Grid Search

Run the full 17-combo grid search on AAPL's own inner train (first 70% of its minutes) and pick the best-performing combo.

### Decision

Whichever route produces higher improvement on AAPL's inner train is selected. Both routes are reported for transparency.

**Based on the synthetic test:** MSFT is AAPL's nearest neighbour (both are approximately $30, penny-spread, high-frequency stocks). Route A would transfer Ensemble(min_votes=3, deadline_frac=0.7). Route B may confirm this or find a different configuration depending on the real AAPL data.

---

## 9. Results and Interpretation

### Reading the Results Table

```
Stock  |  Strategy  |  Train %  |  Holdout %  |  Retention  |  Test %
```

- **Train %** — improvement over TWAP on the 189-minute inner training window. This is what the grid search optimised for.
- **Holdout %** — improvement over TWAP on the 81-minute held-out window. The algorithm had not seen this data.
- **Retention** — `Holdout / Train`. How much of the in-sample edge survived to the unseen period.
- **Test %** — improvement over TWAP on the separate test CSV (available Monday). This is the final, truly out-of-sample number.

### What the Buy/Sell Alpha Numbers Mean

`Buy α = 2.68 bps` for AMZN means: on average, the algorithm bought AMZN at a price 2.68 basis points below where TWAP would have bought. On a $220 stock, 2.68 bps = **~$0.059 per share**. Across hundreds of minutes per day, that compounds significantly.

`Sell α = 2.58 bps` means: sells received 2.58 bps more than TWAP. Both sides are positive — the strategy is finding real improvement on both the buy and sell leg.

### TWAP Cost vs. Algo Cost

The `TWAP Cost` and `Algo Cost` columns show the raw difference between total buy prices and total sell prices across all minutes:

```
AMZN TWAP Cost = $37.50     Algo Cost = $5.82
```

TWAP paid $37.50 more to buy than it received from selling (net cost of crossing the spread repeatedly). The Ensemble strategy reduced that to $5.82 — an 84% reduction in net execution cost.

For INTC and MSFT (penny-spread stocks), both costs go **negative** — the algo actually receives *more* from sells than it pays for buys, i.e., it earns the spread rather than paying it. This is the algorithmic equivalent of providing liquidity rather than consuming it.

### Why Ensemble Won Everywhere

The three signals are weakly correlated with each other. Spread tightening, OFI directionality, and microprice tilt each pick up different aspects of intra-minute LOB dynamics. Requiring two or three to agree filters out noise and concentrates execution into moments where multiple indicators simultaneously point in the right direction. The price-improvement gate provides a hard floor: the strategy can never do worse than TWAP on any individual minute.

---

## 10. Presentation Suite — All Figures Explained

`make_presentation.py` generates a 9-figure deck covering the full project arc from Milestone 1 to V9 Final. Run it with:

```python
from v9_final.make_presentation import run_presentation

run_presentation(
    train_dir   = "./1. Project Files",
    project_dir = "./1. Project Files",
    out_dir     = "./1. Project Files/v9_presentation"
)
```

### Figure Guide

**p01 — Alpha Journey (the headline slide)**

Portfolio overall α (bps) as a bar chart across every version of the algorithm — V5 In-Sample, V5 OOS, V6 In-Sample, V7 Dev, V7 OOS, V8 OOS, V9 Train, V9 Holdout. Hatched bars are out-of-sample or holdout periods, solid bars are in-sample.

*What to say:* Every version from V5 through V8 produced negative portfolio alpha — the algorithm was losing to the TWAP benchmark even in-sample for most stocks. V9 is the first version to show positive alpha across the whole portfolio, and that alpha survives to the held-out period.

**p02 — Buy vs Sell Alpha Decomposition**

The same version progression broken down into buy-leg alpha and sell-leg alpha separately. This reveals *why* earlier versions struggled: the sell side was consistently dragging the portfolio into negative territory (large negative sell α), even when the buy side showed small improvements.

*What to say:* V9's key architectural advance is that it handles both legs symmetrically — the Ensemble fires for SELL orders using the same three signals, mirrored. That is why sell α turns positive for the first time.

**p03 — Per-Stock Progression**

Line chart showing each individual stock's overall α across all versions. AMZN and GOOG (wider spread stocks) had consistently smaller magnitudes — there is more room to improve when the spread is wider. INTC and MSFT (penny-spread, ultra-high-frequency) were near zero in early versions despite having the most signal-rich LOBs — the old strategies were unable to exploit sub-cent within-minute timing.

**p04 — Walk-Forward Validation**

Train vs holdout improvement (%) bars per stock, with retention ratio overlaid as a line. This is the direct answer to the professors' Milestone 1 critique. The original submission had no holdout evaluation — all numbers were in-sample.

*What to say:* We deliberately designed V9 to be transparent about this. The retention ratios (91%, 97%, 64%, 66%) show where the edge is robust and where there is meaningful decay. None fall below 50%, which would indicate severe overfitting.

**p05 — Retention Flags**

A standalone retention ratio bar chart with colour-coded zones. Green (>70%) = edge is robust. Amber (50–70%) = edge present but with decay, worth monitoring. Red (<50%) = overfitting concern. AMZN and GOOG are green; INTC and MSFT are amber.

**p06 — Improvement Distributions**

Box plots of per-minute buy improvement (bps) and sell improvement (bps) for each stock. The distribution shape is characteristic: a large spike at zero (the algorithm fell back to TWAP that minute — no condition was met) with a positive tail (the Ensemble fired and found a better price). There are no negative values — the price-improvement gate ensures the algo never does worse than TWAP on any individual minute.

**p07 — Descriptive Statistics Tables (professor-format)**

The raw `describe()` output for buy and sell improvement ($) per minute, per stock, rendered as a formatted figure. Equivalent to calling `series.describe()` in Python. The 50th percentile (median) being zero confirms that the algorithm executes at TWAP in the majority of minutes, and the edge comes from the minority of minutes where it finds a better tick.

**p08 — TWAP vs Algo Comparison Table**

The formatted comparison table per stock:

| Metric | TWAP | Algo (V9 Ensemble) | Improvement ($) | Improvement (bps) |
|--------|------|--------------------|-----------------|-------------------|
| Average Buy Price | $223.51937 | $223.45959 | +$0.05978 | +2.68 bps |
| Average Sell Price | $223.38048 | $223.43804 | +$0.05756 | +2.58 bps |
| Avg BUY-SELL Spread | $0.13889 | $0.02156 | +$0.11733 | +5.25 bps |

The spread row is particularly striking: the algo's effective spread (difference between its average buy and sell prices) collapses to near zero because it is buying at below-mid and selling at above-mid prices by timing within the minute.

**p09 — V9 Buy/Sell Alpha per Stock (Train vs Holdout)**

Grouped bars showing buy α and sell α for each stock in two periods — full train (solid) and holdout (hatched). Confirms that both the buy and sell improvements are present in both periods, not just in the training window.

---

## 11. Per-Minute Improvement Statistics

These are the actual numbers produced by the V9 Ensemble on the real training data (270 minutes per stock). The `describe()` output is printed to the terminal every time `run_presentation()` is called, and rendered as `p07_describe_tables.png`.

### AMZN (spread ~$0.14, ~650 ticks/min)

```
BUY IMPROVEMENT ($)
count    270.000000
mean       0.059778       ← paid $0.060 less per share than TWAP on average
std        0.089820
min        0.000000       ← worst case = TWAP (never worse)
25%        0.000000       ← algo fell back to TWAP in bottom 25% of minutes
50%        0.020000       ← median minute: saved $0.02 vs TWAP
75%        0.090000
max        0.600000       ← best minute: saved $0.60 vs TWAP

SELL IMPROVEMENT ($)
count    270.000000
mean       0.057556       ← received $0.058 more per share than TWAP on average
std        0.105192
min        0.000000
25%        0.000000
50%        0.010000
75%        0.080000
max        1.070000

TWAP vs ALGO COMPARISON
Average Buy Price    TWAP=$223.51937   Algo=$223.45959   Improvement=+$0.05978 (+2.68 bps)
Average Sell Price   TWAP=$223.38048   Algo=$223.43804   Improvement=+$0.05756 (+2.58 bps)
Avg BUY-SELL Spread  TWAP=$0.13889    Algo=$0.02156     Improvement=+$0.11733 (+5.25 bps)
```

### GOOG (spread ~$0.29, ~370 ticks/min)

```
BUY IMPROVEMENT ($)     mean=$0.119 (+2.09 bps)    max=$1.080
SELL IMPROVEMENT ($)    mean=$0.100 (+1.75 bps)    max=$1.200
Avg BUY-SELL Spread     TWAP=$0.293  →  Algo=$0.073  (reduction of $0.220 / 3.84 bps)
```

### INTC (spread ~$0.01, ~1,600 ticks/min)

```
BUY IMPROVEMENT ($)     mean=$0.0074 (+2.73 bps)   max=$0.090
SELL IMPROVEMENT ($)    mean=$0.0050 (+1.83 bps)   max=$0.040
Avg BUY-SELL Spread     TWAP=$0.0102  →  Algo=-$0.002  (algo earns the spread)
```

Note: a negative average algo spread for INTC/MSFT means the algo consistently buys below mid and sells above mid — it is *providing* rather than consuming liquidity within the minute, effectively earning the spread rather than paying it.

### MSFT (spread ~$0.01, ~1,640 ticks/min)

```
BUY IMPROVEMENT ($)     mean=$0.0085 (+2.76 bps)   max=$0.090
SELL IMPROVEMENT ($)    mean=$0.0067 (+2.20 bps)   max=$0.060
Avg BUY-SELL Spread     TWAP=$0.0102  →  Algo=-$0.005  (algo earns the spread)
```

### Reading these numbers in context

The zero 25th percentile across all stocks confirms the strategy's design: it never forces execution at a suboptimal tick. In the majority of minutes the LOB conditions never triggered all three Ensemble signals simultaneously, so the algorithm defaulted to TWAP (the safe fallback). The edge is concentrated in the minority of minutes where the book offered a clean, multi-signal opportunity — and in those minutes the improvement can be substantial (up to $0.60 on AMZN, up to $1.20 on GOOG).

---

## Monday Workflow

When test CSVs arrive (and/or AAPL training data), run:

```powershell
# In PowerShell, from the repo root
$env:PYTHONUTF8 = "1"
python -m v9_final.run_all `
    --train-dir "./1. Project Files" `
    --test-dir  "<path to test CSVs>"
```

This will:
1. Re-tune each of the 4 known stocks on their inner train (same parameters will be re-selected — this is deterministic)
2. Evaluate on holdout and test
3. Run AAPL autonomous selection if `AAPL_5levels_train.csv` is present in `--train-dir`
4. Print the full results table with the `Test Improv (%)` column populated
5. Save `v9_results_table.csv` to `./1. Project Files/`

To regenerate all plots with the test column visible:

```python
from v9_final.run_all import run_full_pipeline, build_results_table
from v9_final.make_plots import make_all_plots

pipe  = run_full_pipeline("./1. Project Files", test_dir="<test dir>")
table = build_results_table(pipe)
make_all_plots(pipe, table, "./1. Project Files/v9_plots")
```

**If any test improvement is negative:** the algo lost to TWAP on the test day for that stock. This does not invalidate the methodology — a single day can be anomalous (earnings, macro event, unusual spread regime). The holdout retention ratios remain the primary evidence of out-of-sample validity. Check whether the test CSV has the same column structure as the training data before drawing conclusions.

---

## Code Map — Where to Find Everything

| Question | File | Function |
|---------|------|---------|
| How does the data load? | `backtest.py` | `load_lob()` |
| How is TWAP computed? | `strategies.py` | `TWAP.execute()` |
| How is OFI computed? | `strategies.py` | `_compute_ofi_array()` |
| How is microprice computed? | `strategies.py` | `_microprice()` |
| How are rolling thresholds built? | `backtest.py` | `precompute_context()` |
| How does the grid search work? | `meta.py` | `tune_strategy_on_train()` |
| How does AAPL routing work? | `meta.py` | `autonomous_aapl_selection()` |
| How is the project metric computed? | `backtest.py` | `project_metric()` |
| How is the 70/30 split done? | `backtest.py` | `chronological_split()` |
| How are execution detail plots generated? | `make_plots.py` | `make_all_plots()` |
| How is the presentation suite generated? | `make_presentation.py` | `run_presentation()` |
| How are historical milestones loaded? | `make_presentation.py` | `load_milestone_data()` |
| How are per-minute stats computed? | `make_presentation.py` | `compute_per_minute_stats()` |
| How does the CLI pipeline work? | `run_all.py` | `run_full_pipeline()` |
