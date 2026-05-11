"""
Generate LOBSTER-format CSVs for smoke-testing the v9 pipeline.
Mean-reverting within-minute dynamics give timing signals to strategies.
"""
import os
import numpy as np
import pandas as pd

PROFILES = {
    "AMZN": {"price": 220.0,  "spread": 0.14, "tpm": 60},
    "GOOG": {"price": 572.0,  "spread": 0.29, "tpm": 50},
    "INTC": {"price":  27.0,  "spread": 0.01, "tpm": 110},
    "MSFT": {"price":  30.0,  "spread": 0.01, "tpm": 130},
    "AAPL": {"price": 180.0,  "spread": 0.02, "tpm": 100},
}

_SESSION_START = 9 * 3600 + 30 * 60   # 09:30:00 in seconds


def _fmt_time(total_sec):
    hh = int(total_sec // 3600)
    mm = int((total_sec % 3600) // 60)
    ss = total_sec % 60.0
    return f"{hh:02d}:{mm:02d}:{ss:09.6f}"


def generate_stock(profile, n_minutes=200, seed=42):
    rng    = np.random.default_rng(seed)
    price  = profile["price"]
    spread = profile["spread"]
    tpm    = profile["tpm"]

    rows = []
    mid  = price

    for m in range(n_minutes):
        # Minute-level random walk
        mid += rng.normal(0, price * 0.001)
        mid_start    = mid
        minute_drift = rng.normal(0, price * 0.0005)

        n_ticks = max(5, int(rng.poisson(tpm)))
        offsets  = np.sort(rng.uniform(0, 59.9, n_ticks))

        for t_idx, t_off in enumerate(offsets):
            # Mean-reverting intra-minute path
            frac     = t_idx / max(n_ticks - 1, 1)
            revert   = (mid_start + minute_drift * frac - mid) * 0.35
            mid     += revert + rng.normal(0, price * 0.00015)

            half_sp = spread / 2.0
            bid     = round(mid - half_sp, 4)
            ask     = round(mid + half_sp, 4)
            sp      = round(ask - bid, 4)

            base_sz = max(1, int(rng.exponential(100)))

            bids_p = [round(bid - i * spread * 0.5, 4) for i in range(5)]
            asks_p = [round(ask + i * spread * 0.5, 4) for i in range(5)]
            bids_s = [max(1, int(rng.exponential(base_sz))) for _ in range(5)]
            asks_s = [max(1, int(rng.exponential(base_sz))) for _ in range(5)]

            total_sec = _SESSION_START + m * 60 + t_off

            rows.append({
                "Time":                        _fmt_time(total_sec),
                "BidPrice_5":                  bids_p[4],
                "BidPrice_4":                  bids_p[3],
                "BidPrice_3":                  bids_p[2],
                "BidPrice_2":                  bids_p[1],
                "BidPrice_1":                  bids_p[0],
                "BidSize_5":                   bids_s[4],
                "BidSize_4":                   bids_s[3],
                "BidSize_3":                   bids_s[2],
                "BidSize_2":                   bids_s[1],
                "BidSize_1":                   bids_s[0],
                "AskPrice_1":                  asks_p[0],
                "AskPrice_2":                  asks_p[1],
                "AskPrice_3":                  asks_p[2],
                "AskPrice_4":                  asks_p[3],
                "AskPrice_5":                  asks_p[4],
                "AskSize_1":                   asks_s[0],
                "AskSize_2":                   asks_s[1],
                "AskSize_3":                   asks_s[2],
                "AskSize_4":                   asks_s[3],
                "AskSize_5":                   asks_s[4],
                "OrderID":                     int(rng.integers(1_000_000, 9_999_999)),
                "Size":                        base_sz,
                "Price":                       round(mid, 4),
                "Direction_1=Buy_-1=Sell":     1 if rng.random() > 0.5 else -1,
                "NewLimitOrder_1=Yes_0=No":    1 if rng.random() > 0.7 else 0,
                "PartialCancel_1=Yes_0=No":    0,
                "FullDelete_1=Yes_0=No":       0,
                "VisibleExecution_1=Yes_0=No": 1 if rng.random() > 0.8 else 0,
                "HiddenExecution_1=Yes_0=No":  0,
                "TradingHalt_1=Yes_0=No":      0,
                "Spread":                      sp,
                "MidPrice":                    round(mid, 4),
            })

    return pd.DataFrame(rows)


def generate_all(out_dir, n_minutes=200):
    os.makedirs(out_dir, exist_ok=True)
    for i, (stock, profile) in enumerate(PROFILES.items()):
        df   = generate_stock(profile, n_minutes=n_minutes, seed=42 + i)
        path = os.path.join(out_dir, f"{stock}_5levels_train.csv")
        df.to_csv(path, index=False)
        print(f"  {stock}: {len(df):,} rows  ->  {path}")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "./v9_final/data"
    generate_all(out)
