# v9 Final — Market Microstructure Execution Algorithm

**Course project — Final Submission**

This folder contains the complete, self-contained v9 execution-algorithm pipeline. It is isolated from all prior v6/v7/v8 exploratory work and is the only version that should be referenced for the final presentation.

---

## What This Does

We are given a limit order book (LOB) data feed for five stocks. The task is to execute **exactly 1 share per minute** — our algorithm chooses *which tick* within each minute to execute on, aiming to beat a naive TWAP benchmark that always executes at the very first tick of the minute.

The project metric is:

```
Improvement (%) = 100 × (1 − Algo Cost / TWAP Cost)

where Cost = Total Buy Price − Total Sell Price (across all minutes)
```

A positive improvement means the algorithm paid less to buy and received more from selling than TWAP would have — that is the edge.

---

## Results (Real Training Data)

Four stocks, 270 minutes each. Inner 70% used for tuning; outer 30% as unseen holdout to validate the edge carries forward.

| Stock | Strategy Selected | Train Improvement | Holdout Improvement | Retention |
|-------|------------------|:-----------------:|:-------------------:|:---------:|
| AMZN  | Ensemble (3-vote, 70% deadline) | **86.7%** | **78.6%** | 90.7% ✓ |
| GOOG  | Ensemble (2-vote, 80% deadline) | **75.7%** | **73.7%** | 97.4% ✓ |
| INTC  | Ensemble (3-vote, 70% deadline) | **135.4%** | **86.9%** | 64.2% ~ |
| MSFT  | Ensemble (3-vote, 70% deadline) | **166.0%** | **109.9%** | 66.2% ~ |
| AAPL  | Autonomous (pending) | — | — | — |

`✓` Robust (>70% retention)  `~` Caution (50–70% retention)

No stock falls below 50% retention — the walk-forward criticism raised after Milestone 1 is addressed with hard numbers.

**Test data column will auto-populate when test CSVs arrive Monday** — see [WALKTHROUGH.md](WALKTHROUGH.md#monday-workflow) for the one-command workflow.

---

## Repository Structure

```
v9_final/
├── README.md                  ← you are here
├── WALKTHROUGH.md             ← full explanation of strategy, math, and results
├── RESULTS_SUMMARY.md         ← generated results report (auto-updated on each run)
│
├── strategies.py              ← 6 strategy classes + parameter grids
├── backtest.py                ← data loading, backtesting engine, metric computation
├── meta.py                    ← fingerprinting, grid-search tuner, AAPL autonomous routing
├── run_all.py                 ← master CLI pipeline (entry point)
├── make_plots.py              ← per-stock execution plots (5 figures)
├── make_presentation.py       ← full presentation suite (9 figures + console tables)
├── make_synthetic_data.py     ← synthetic LOB generator for smoke-testing
└── __init__.py                ← package exports
```

**Downstream artifacts** (written to `./1. Project Files/`):
```
1. Project Files/
├── v9_results_table.csv           ← per-stock metrics table
├── v9_plots/                      ← per-stock execution detail plots
│   ├── p1_results_bar.png
│   ├── p2_strategy_distribution.png
│   ├── p3_buy_sell_alpha.png
│   ├── p4_aapl_routing.png        ← appears when AAPL data is present
│   └── p5_<STOCK>_execution_curves.png  (× 4)
└── v9_presentation/               ← full presentation suite (9 figures)
    ├── p01_alpha_journey.png          ← portfolio α evolution V5 → V9
    ├── p02_buysell_decomposition.png  ← buy vs sell α per version
    ├── p03_perstock_progression.png   ← per-stock α across all versions
    ├── p04_walkforward_validation.png ← train vs holdout with retention ratios
    ├── p05_retention_flags.png        ← retention ratio bar with risk zones
    ├── p06_improvement_distributions.png ← per-minute improvement box plots
    ├── p07_describe_tables.png        ← professor-format describe() tables
    ├── p08_twap_comparison.png        ← TWAP vs Algo comparison table
    └── p09_v9_alpha_per_stock.png     ← buy/sell α train vs holdout per stock
```

---

## Setup

**Requirements:** Python 3.9+ with `numpy`, `pandas`, `matplotlib`.

```bash
pip install numpy pandas matplotlib
```

**Run the full pipeline on real training data:**

```bash
# Windows PowerShell
$env:PYTHONUTF8 = "1"
python -m v9_final.run_all --train-dir "./1. Project Files"
```

**Run with test data when it arrives Monday:**

```bash
$env:PYTHONUTF8 = "1"
python -m v9_final.run_all --train-dir "./1. Project Files" --test-dir "<path-to-test-csvs>"
```

**Smoke test on synthetic data first (no real CSVs needed):**

```bash
$env:PYTHONUTF8 = "1"
python -m v9_final.run_all --synthetic --train-dir ./v9_final/data
```

**Generate per-stock execution plots:**

```python
from v9_final.run_all import run_full_pipeline, build_results_table
from v9_final.make_plots import make_all_plots

pipe  = run_full_pipeline("./1. Project Files")
table = build_results_table(pipe)
make_all_plots(pipe, table, "./1. Project Files/v9_plots")
```

**Generate the full presentation suite (9 figures + console tables):**

```python
from v9_final.make_presentation import run_presentation

run_presentation(
    train_dir   = "./1. Project Files",   # where the training CSVs live
    project_dir = "./1. Project Files",   # where the milestone CSVs live (same dir)
    out_dir     = "./1. Project Files/v9_presentation"
)
```

This prints the professor-format `describe()` tables to the terminal and saves all 9 presentation figures. See [Presentation Plots](#presentation-plots) below for what each figure shows.

---

## The Six Strategies

| # | Name | Core Idea |
|---|------|-----------|
| 0 | **TWAP** | Baseline — always executes at first tick of each minute |
| 1 | **SpreadQuantile** | Wait for a tick where the bid-ask spread is unusually tight |
| 2 | **OFIContrarian** | Use order-flow imbalance to time against short-term pressure |
| 3 | **Microprice** | Use the depth-weighted mid price to detect book imbalance |
| 4 | **AdaptiveDeadline** | Hunt for a price improvement early; accept any gain after 70% of ticks elapsed |
| 5 | **Ensemble** | Combine all three signals with a voting mechanism; only fire when multiple agree |

See [WALKTHROUGH.md](WALKTHROUGH.md) for the full intuition, math, and diagrams for each strategy.

---

## Presentation Plots

`make_presentation.py` generates a self-contained 9-figure deck that covers the **full project arc** — from Milestone 1 through V9 Final — and includes every metric the professors asked about.

| Figure | File | What it shows |
|--------|------|---------------|
| 1 | `p01_alpha_journey.png` | Portfolio overall α (bps) across every version (V5 IS → V5 OOS → V6 → V7 Dev → V7 OOS → V8 OOS → V9 Train → V9 Holdout). Hatched bars = OOS/holdout periods. The key visual story: all prior versions had negative portfolio α; V9 is the first to go positive. |
| 2 | `p02_buysell_decomposition.png` | Buy α vs Sell α side-by-side per version. Shows that earlier strategies hurt the sell leg (large negative sell α); V9 achieves positive α on both legs simultaneously. |
| 3 | `p03_perstock_progression.png` | Per-stock overall α across versions as a line chart. Shows which stocks were consistently hard or easy to improve and where V9 breaks the pattern. |
| 4 | `p04_walkforward_validation.png` | V9 train vs holdout improvement (%) bars per stock with retention ratio overlaid. Directly answers the professors' Milestone 1 critique. |
| 5 | `p05_retention_flags.png` | Standalone retention ratio bar with green/amber/red risk zones (>70% / 50-70% / <50%). No stock below 50%. |
| 6 | `p06_improvement_distributions.png` | Box plots of per-minute buy and sell improvement (bps) per stock. Zero mass = minutes falling back to TWAP; positive tail = successful Ensemble timing. |
| 7 | `p07_describe_tables.png` | Professor-format `describe()` tables rendered as a figure — count, mean, std, min, 25/50/75%, max for buy and sell improvement ($) per stock. |
| 8 | `p08_twap_comparison.png` | Formatted TWAP vs Algo comparison table per stock: Average Buy Price, Average Sell Price, Average BUY-SELL Spread — in dollars and bps, colour-coded green/red. |
| 9 | `p09_v9_alpha_per_stock.png` | Per-stock Buy α and Sell α (full-train solid, holdout hatched) showing the edge is consistent across both sides and both periods. |

**Actual numbers from the real data run:**

| Stock | Avg Buy improvement | Avg Sell improvement | Spread reduction |
|-------|--------------------:|---------------------:|-----------------:|
| AMZN  | $0.060 / 2.68 bps   | $0.058 / 2.58 bps    | $0.117 / 5.25 bps |
| GOOG  | $0.119 / 2.09 bps   | $0.100 / 1.75 bps    | $0.220 / 3.84 bps |
| INTC  | $0.007 / 2.73 bps   | $0.005 / 1.83 bps    | $0.012 / 4.55 bps |
| MSFT  | $0.008 / 2.76 bps   | $0.007 / 2.20 bps    | $0.015 / 4.96 bps |

---

## Key Design Decisions

- **No lookahead bias.** Rolling statistics (spread quantile, OFI threshold) use `shift(1)` before rolling — they see only past data at each tick.
- **Chronological split, not random.** Holdout is the last 30% of minutes in time order, as it would be in practice.
- **Retention ratio.** We explicitly compute `holdout_improvement / train_improvement` and flag anything under 50% as a potential overfit.
- **AAPL is fully autonomous.** The algorithm has never seen AAPL data. On Monday it will fingerprint AAPL's microstructure and choose its strategy without human input via two competing routes (nearest-neighbour transfer vs. direct grid search).

---

## Contact / Questions

See [WALKTHROUGH.md](WALKTHROUGH.md) for a complete explanation of every component. For code-level questions, each module has inline comments at non-obvious decision points.
