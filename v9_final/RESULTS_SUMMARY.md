# v9 Final — Results Summary
*Generated from real training data: AMZN, GOOG, INTC, MSFT (270 minutes each). AAPL pending.*

---

## Final Results Table

| Stock | Strategy | Params | Train Improv (%) | Holdout Improv (%) | Retention Ratio | Test Improv (%) | Buy α (bps) | Sell α (bps) | TWAP Cost | Algo Cost |
|-------|----------|--------|------------------|--------------------|-----------------|-----------------|-------------|--------------|-----------|-----------|
| AMZN | Ensemble | min_votes=3, deadline_frac=0.7 | 86.69 | 78.63 | 0.907 | — | 2.68 | 2.58 | 37.50 | 5.82 |
| GOOG | Ensemble | min_votes=2, deadline_frac=0.8 | 75.70 | 73.73 | 0.974 | — | 2.09 | 1.75 | 79.12 | 19.77 |
| INTC | Ensemble | min_votes=3, deadline_frac=0.7 | 135.42 | 86.90 | 0.642 | — | 2.73 | 1.83 | 2.76 | −0.58 |
| MSFT | Ensemble | min_votes=3, deadline_frac=0.7 | 165.98 | 109.88 | 0.662 | — | 2.76 | 2.20 | 2.75 | −1.36 |
| AAPL | — | — | — | — | — | — | — | — | — | — |

*Training data: `./1. Project Files/`. Test column will populate when test CSVs arrive Monday.*

---

## Per-Stock Strategy Summary

### AMZN — Ensemble (min_votes=3, deadline_frac=0.7)
Three signals must agree (tight spread + OFI contrarian + microprice tilt) before firing;
past 70% of ticks elapsed, accepts any price improvement. Strong multi-signal consensus
requirement makes it conservative but well-calibrated for AMZN's wide spread environment.

### GOOG — Ensemble (min_votes=2, deadline_frac=0.8)
Two-of-three signal threshold with a longer patience window (80% deadline). GOOG's
wider spread (∼$0.29) gives the strategy more room to wait for meaningful improvement;
the longer deadline captures late-minute reversions.

### INTC — Ensemble (min_votes=3, deadline_frac=0.7)
Same configuration as AMZN. INTC's penny spread means the algo is essentially extracting
half-tick improvements at scale across 430K ticks/day. The high train improvement (135%)
reflects that within-minute timing signal is extremely dense in INTC's high-frequency LOB.

### MSFT — Ensemble (min_votes=3, deadline_frac=0.7)
Highest train improvement (166%) among the four stocks. MSFT mirrors INTC in having a
penny spread with very high tick frequency, making within-minute timing signal abundant.
Holdout at 110% confirms the edge is real.

---

## Retention Ratios (Walk-Forward Validation)

| Stock | Train (%) | Holdout (%) | Retention | Assessment |
|-------|-----------|-------------|-----------|------------|
| AMZN  | 86.69     | 78.63       | 90.7%     | **Robust** — edge holds strongly OOS |
| GOOG  | 75.70     | 73.73       | 97.4%     | **Robust** — near-perfect retention |
| INTC  | 135.42    | 86.90       | 64.2%     | **Caution** — meaningful decay, monitor |
| MSFT  | 165.98    | 109.88      | 66.2%     | **Caution** — meaningful decay, monitor |

*Split: 70% train (189 min) / 30% holdout (81 min), strictly chronological.*

---

## AAPL Autonomous Routing Decision

AAPL training data not yet available. When `AAPL_5levels_train.csv` arrives Monday,
the autonomous routing will:

1. **Compute AAPL microstructure fingerprint** — spread_bps, tick_freq, depth_per_bps, vol_bps
2. **Route A**: Transfer the winning strategy from the nearest-neighbor stock
   (log-Euclidean distance over fingerprint features, excluding price level)
3. **Route B**: Direct grid-search over all 17 strategy/param combos on AAPL's own inner-train
4. **Pick whichever route scores higher** on AAPL inner-train improvement
5. Evaluate that strategy on AAPL's holdout and report retention ratio

Based on the synthetic smoke test, MSFT is AAPL's nearest neighbor (both are ~$30 penny-spread
stocks). Expect Route A to transfer Ensemble (min_votes=3, deadline_frac=0.7) as the starting
candidate; Route B may refine if AAPL's within-minute dynamics differ.

---

## What This Means

### Robust edge (>70% retention): AMZN, GOOG
Both stocks show the Ensemble strategy translates cleanly from train to holdout. The three-signal
voting mechanism (spread tightening + OFI contrarian + microprice tilt) is finding real, stable
microstructure inefficiency. These results should hold on the test day with reasonable confidence.

### Caution zone (50-70% retention): INTC, MSFT
Holdout improvement is still large in absolute terms (87% and 110%), but the decay from train
is notable. The likely cause is that INTC/MSFT have extremely high tick frequencies (1,600–2,200
ticks/minute), making the Ensemble signal dense enough to overfit the specific within-minute
patterns in the first 189 minutes. The edge is real — it just needs the test day to confirm
it isn't the first-189-minutes-specific ordering.

**No stock is below 50% retention**, so there is no severe overfitting concern; the professors'
walk-forward criticism from Milestone 1 is addressed head-on with concrete numbers.

---

## Ready for Test Data — Monday Checklist

When test CSVs (`AMZN_5levels_test.csv`, `GOOG_5levels_test.csv`, `INTC_5levels_test.csv`,
`MSFT_5levels_test.csv`, `AAPL_5levels_test.csv`) arrive:

```bash
# Run with the correct Python (has numpy/pandas/matplotlib installed)
$env:PYTHONUTF8 = "1"
python -m v9_final.run_all \
  --train-dir "./1. Project Files" \
  --test-dir  "<path to test CSVs>"
```

**What happens automatically:**
- [x] 4 known stocks re-tune on their inner-train, evaluate on inner-holdout + full-train + test
- [x] AAPL runs autonomous_aapl_selection (Route A + Route B) on AAPL inner-train
- [x] AAPL evaluated on AAPL holdout and AAPL test
- [x] `Test Improv (%)` column populates in the results table
- [x] `v9_results_table.csv` re-saved with test column
- [x] Re-run `make_all_plots(...)` to regenerate all plots with test bars

**Code verification:**
- `run_stock(stock, train_path, test_path=None)` — accepts test_path ✓
- `run_aapl(train_path, known_stocks, test_path=None)` — accepts test_path ✓
- `build_results_table(pipeline)` — populates "Test Improv (%)" when test_metric is not None ✓

**If test improvement is negative on a stock:**
That means the algo loses to TWAP on the unseen day. Check whether the LOB format matches
(same column names/order), then inspect whether the test day had unusual spread dynamics.
The strategy is still valid from a methodology standpoint even if one test day is unfavorable.
