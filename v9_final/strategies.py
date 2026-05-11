import numpy as np
import pandas as pd


def _exec_col(side):
    return "AskPrice_1" if side == "BUY" else "BidPrice_1"


def _compute_ofi_array(group):
    """Cont/Kukanov/Stoikov 2014 cumulative OFI, tick-by-tick within one minute."""
    bids = group["BidPrice_1"].values
    asks = group["AskPrice_1"].values
    bsz  = group["BidSize_1"].values
    asz  = group["AskSize_1"].values

    ofi = np.zeros(len(group))
    for i in range(1, len(group)):
        if bids[i] > bids[i - 1]:
            bc = bsz[i]
        elif bids[i] == bids[i - 1]:
            bc = bsz[i] - bsz[i - 1]
        else:
            bc = -bsz[i]

        if asks[i] < asks[i - 1]:
            ac = asz[i]
        elif asks[i] == asks[i - 1]:
            ac = asz[i] - asz[i - 1]
        else:
            ac = -asz[i]

        ofi[i] = ofi[i - 1] + (bc - ac)
    return ofi


def _microprice(row):
    bp, ap = row["BidPrice_1"], row["AskPrice_1"]
    bs, as_ = row["BidSize_1"], row["AskSize_1"]
    denom = bs + as_
    if denom == 0:
        return (bp + ap) / 2.0
    return (bp * as_ + ap * bs) / denom


# ── Base ─────────────────────────────────────────────────────────────────────

class Strategy:
    name = "Base"

    def execute(self, group, side, ctx):
        raise NotImplementedError


# ── 1. TWAP ──────────────────────────────────────────────────────────────────

class TWAP(Strategy):
    name = "TWAP"

    def execute(self, group, side, ctx):
        return group.iloc[0][_exec_col(side)]


# ── 2. SpreadQuantile ─────────────────────────────────────────────────────────

class SpreadQuantile(Strategy):
    name = "SpreadQuantile"

    def __init__(self, rolling_window=500, q=0.30):
        self.rolling_window = rolling_window
        self.q = q

    def execute(self, group, side, ctx):
        col = _exec_col(side)
        rst = ctx.get("rolling_spread_threshold")

        # Build per-tick threshold array, handling duplicate timestamps
        if rst is not None:
            mask = rst.index.isin(group.index)
            vals = rst.loc[mask].values
            if len(vals) == len(group):
                thresh_arr = vals
            elif len(vals) > 0:
                thresh_arr = np.full(len(group), np.median(vals))
            else:
                thresh_arr = np.full(len(group), group["Spread"].median())
        else:
            thresh_arr = np.full(len(group), group["Spread"].median())

        spreads = group["Spread"].values
        prices  = group[col].values

        for i in range(len(group)):
            if spreads[i] <= thresh_arr[i]:
                return prices[i]

        return prices[-1]  # fallback: last tick


# ── 3. OFIContrarian ──────────────────────────────────────────────────────────

class OFIContrarian(Strategy):
    name = "OFIContrarian"

    def __init__(self, threshold_pct=0.70, rolling_window=60, fallback="first"):
        self.threshold_pct = threshold_pct
        self.rolling_window = rolling_window
        self.fallback = fallback

    def execute(self, group, side, ctx):
        col     = _exec_col(side)
        ofi_arr = _compute_ofi_array(group)

        ofi_fn = ctx.get("ofi_threshold_for_minute")
        if callable(ofi_fn):
            try:
                minute = group.index[0].floor("min")
                thresh = ofi_fn(minute)
            except Exception:
                thresh = np.percentile(np.abs(ofi_arr) + 1e-9, self.threshold_pct * 100)
        else:
            thresh = np.percentile(np.abs(ofi_arr) + 1e-9, self.threshold_pct * 100)

        prices = group[col].values
        for i in range(len(group)):
            cum = ofi_arr[i]
            if side == "BUY"  and cum < -thresh:
                return prices[i]
            if side == "SELL" and cum >  thresh:
                return prices[i]

        return prices[0]  # fallback: first tick (TWAP-equivalent)


# ── 4. Microprice ─────────────────────────────────────────────────────────────

class Microprice(Strategy):
    name = "Microprice"

    def __init__(self, alpha=0.10):
        self.alpha = alpha

    def execute(self, group, side, ctx):
        col    = _exec_col(side)
        prices = group[col].values

        for i, (_, row) in enumerate(group.iterrows()):
            mp     = _microprice(row)
            ask    = row["AskPrice_1"]
            bid    = row["BidPrice_1"]
            spread = row["Spread"]

            if side == "BUY"  and mp >= ask - self.alpha * spread:
                return prices[i]
            if side == "SELL" and mp <= bid + self.alpha * spread:
                return prices[i]

        return prices[0]  # fallback: first tick


# ── 5. AdaptiveDeadline ───────────────────────────────────────────────────────

class AdaptiveDeadline(Strategy):
    name = "AdaptiveDeadline"

    def __init__(self, improvement_bps=1.0, deadline_frac=0.7):
        self.improvement_bps = improvement_bps
        self.deadline_frac   = deadline_frac

    def execute(self, group, side, ctx):
        col   = _exec_col(side)
        n     = len(group)
        prices = group[col].values

        first_price  = prices[0]
        improv_delta = first_price * self.improvement_bps / 1e4
        deadline_idx = int(n * self.deadline_frac)

        best_so_far = first_price

        for i in range(n):
            p = prices[i]
            if side == "BUY":
                best_so_far = min(best_so_far, p)
                if i < deadline_idx:
                    if p < first_price - improv_delta:
                        return p
                else:
                    if p < first_price:
                        return p
            else:
                best_so_far = max(best_so_far, p)
                if i < deadline_idx:
                    if p > first_price + improv_delta:
                        return p
                else:
                    if p > first_price:
                        return p

        return first_price  # fallback: first tick


# ── 6. Ensemble ───────────────────────────────────────────────────────────────

class Ensemble(Strategy):
    name = "Ensemble"

    def __init__(self, min_votes=2, improvement_bps=1.0, deadline_frac=0.7,
                 spread_q=0.30, alpha=0.10):
        self.min_votes       = min_votes
        self.improvement_bps = improvement_bps
        self.deadline_frac   = deadline_frac
        self.spread_q        = spread_q
        self.alpha           = alpha

    def execute(self, group, side, ctx):
        col   = _exec_col(side)
        n     = len(group)
        prices = group[col].values

        first_price  = prices[0]
        improv_delta = first_price * self.improvement_bps / 1e4
        deadline_idx = int(n * self.deadline_frac)

        # Pre-compute OFI and threshold
        ofi_arr = _compute_ofi_array(group)
        ofi_fn  = ctx.get("ofi_threshold_for_minute")
        if callable(ofi_fn):
            try:
                minute    = group.index[0].floor("min")
                ofi_thresh = ofi_fn(minute)
            except Exception:
                ofi_thresh = float(np.std(np.abs(ofi_arr)) + 1e-9)
        else:
            ofi_thresh = float(np.std(np.abs(ofi_arr)) + 1e-9)

        # Rolling spread threshold for this group
        rst = ctx.get("rolling_spread_threshold")
        if rst is not None:
            mask = rst.index.isin(group.index)
            vals = rst.loc[mask].values
            if len(vals) == len(group):
                spread_thresh = vals
            elif len(vals) > 0:
                spread_thresh = np.full(n, np.median(vals))
            else:
                spread_thresh = np.full(n, group["Spread"].median())
        else:
            spread_thresh = np.full(n, group["Spread"].median())

        for i, (_, row) in enumerate(group.iterrows()):
            p      = prices[i]
            votes  = 0
            spread = row["Spread"]
            cum    = ofi_arr[i]
            mp     = _microprice(row)
            ask    = row["AskPrice_1"]
            bid    = row["BidPrice_1"]

            # Signal A: tight spread
            if spread <= spread_thresh[i]:
                votes += 1

            # Signal B: OFI contrarian
            if side == "BUY"  and cum < -ofi_thresh:
                votes += 1
            elif side == "SELL" and cum >  ofi_thresh:
                votes += 1

            # Signal C: microprice tilt
            if side == "BUY"  and mp >= ask - self.alpha * spread:
                votes += 1
            elif side == "SELL" and mp <= bid + self.alpha * spread:
                votes += 1

            past_deadline  = (i >= deadline_idx)
            price_improves = (p < first_price) if side == "BUY" else (p > first_price)

            if votes >= self.min_votes and price_improves:
                return p
            if past_deadline and price_improves:
                return p

        return first_price  # fallback: first tick


# ── Registry ──────────────────────────────────────────────────────────────────

STRATEGY_CLASSES = {
    "TWAP":             TWAP,
    "SpreadQuantile":   SpreadQuantile,
    "OFIContrarian":    OFIContrarian,
    "Microprice":       Microprice,
    "AdaptiveDeadline": AdaptiveDeadline,
    "Ensemble":         Ensemble,
}

PARAM_GRIDS = {
    "SpreadQuantile": [
        {"rolling_window": 200, "q": 0.25},
        {"rolling_window": 200, "q": 0.35},
        {"rolling_window": 500, "q": 0.25},
        {"rolling_window": 500, "q": 0.35},
    ],
    "OFIContrarian": [
        {"threshold_pct": 0.60},
        {"threshold_pct": 0.70},
        {"threshold_pct": 0.80},
    ],
    "Microprice": [
        {"alpha": 0.05},
        {"alpha": 0.10},
        {"alpha": 0.20},
    ],
    "AdaptiveDeadline": [
        {"improvement_bps": 0.5, "deadline_frac": 0.6},
        {"improvement_bps": 0.5, "deadline_frac": 0.8},
        {"improvement_bps": 1.0, "deadline_frac": 0.6},
        {"improvement_bps": 1.0, "deadline_frac": 0.8},
    ],
    "Ensemble": [
        {"min_votes": 2, "deadline_frac": 0.6},
        {"min_votes": 2, "deadline_frac": 0.8},
        {"min_votes": 3, "deadline_frac": 0.7},
    ],
}
