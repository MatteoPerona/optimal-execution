import numpy as np
import pandas as pd

from .strategies import _compute_ofi_array, TWAP


def load_lob(path):
    df = pd.read_csv(path)
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S.%f")
    df = df.set_index("Time")
    df["Minute"] = df.index.floor("min")
    return df


def precompute_context(df, rolling_window=500, spread_q=0.30,
                       ofi_lookback=60, ofi_threshold_pct=0.70):
    # Tick-level rolling spread quantile — shift(1) prevents current-tick lookahead
    rolling_spread_threshold = (
        df["Spread"]
        .shift(1)
        .rolling(window=rolling_window, min_periods=1)
        .quantile(spread_q)
    )

    # Per-minute max-absolute-OFI, then rolling percentile over prior minutes
    unique_minutes = pd.Series(df["Minute"].unique()).sort_values().tolist()

    ofi_by_minute = {}
    for m in unique_minutes:
        grp = df[df["Minute"] == m]
        arr = _compute_ofi_array(grp)
        ofi_by_minute[m] = float(np.abs(arr).max()) if len(arr) > 0 else 0.0

    ofi_thresholds = {}
    for i, m in enumerate(unique_minutes):
        lo = max(0, i - ofi_lookback)
        prior = [ofi_by_minute[unique_minutes[j]] for j in range(lo, i)]
        if prior:
            ofi_thresholds[m] = float(np.percentile(prior, ofi_threshold_pct * 100))
        else:
            ofi_thresholds[m] = 0.0

    def ofi_threshold_for_minute(minute):
        return ofi_thresholds.get(minute, 0.0)

    return {
        "rolling_spread_threshold": rolling_spread_threshold,
        "ofi_threshold_for_minute": ofi_threshold_for_minute,
    }


def run_strategy(df, strategy, ctx):
    buy_prices  = {}
    sell_prices = {}

    for minute, group in df.groupby("Minute"):
        buy_prices[minute]  = strategy.execute(group, "BUY",  ctx)
        sell_prices[minute] = strategy.execute(group, "SELL", ctx)

    buy_series  = pd.Series(buy_prices)
    sell_series = pd.Series(sell_prices)

    total_buy  = float(buy_series.sum())
    total_sell = float(sell_series.sum())
    n_minutes  = len(buy_series)

    return {
        "buy_prices":  buy_series,
        "sell_prices": sell_series,
        "total_buy":   total_buy,
        "total_sell":  total_sell,
        "n_minutes":   n_minutes,
        "avg_buy":     float(buy_series.mean()),
        "avg_sell":    float(sell_series.mean()),
    }


def project_metric(twap_result, algo_result):
    twap_cost = twap_result["total_buy"] - twap_result["total_sell"]
    algo_cost = algo_result["total_buy"] - algo_result["total_sell"]

    if twap_cost == 0:
        improvement_pct = 0.0
    else:
        improvement_pct = 100.0 * (1.0 - algo_cost / twap_cost)

    twap_avg_buy  = twap_result["avg_buy"]
    twap_avg_sell = twap_result["avg_sell"]
    algo_avg_buy  = algo_result["avg_buy"]
    algo_avg_sell = algo_result["avg_sell"]

    mid_proxy = (twap_avg_buy + twap_avg_sell) / 2.0
    if mid_proxy == 0:
        mid_proxy = 1.0

    twap_eff_spread     = twap_avg_buy - twap_avg_sell
    algo_eff_spread     = algo_avg_buy - algo_avg_sell
    twap_eff_spread_bps = twap_eff_spread / mid_proxy * 1e4
    algo_eff_spread_bps = algo_eff_spread / mid_proxy * 1e4
    spread_collapse_bps = twap_eff_spread_bps - algo_eff_spread_bps
    buy_improv_bps      = (twap_avg_buy  - algo_avg_buy)  / mid_proxy * 1e4
    sell_improv_bps     = (algo_avg_sell - twap_avg_sell) / mid_proxy * 1e4

    return {
        "improvement_pct":      improvement_pct,
        "twap_cost":            twap_cost,
        "algo_cost":            algo_cost,
        "twap_eff_spread_bps":  twap_eff_spread_bps,
        "algo_eff_spread_bps":  algo_eff_spread_bps,
        "spread_collapse_bps":  spread_collapse_bps,
        "buy_improv_bps":       buy_improv_bps,
        "sell_improv_bps":      sell_improv_bps,
    }


def chronological_split(df, train_frac=0.7):
    # pd.Series wrapper handles DatetimeArray which lacks .sort_values()
    unique_minutes = pd.Series(df["Minute"].unique()).sort_values()
    n          = len(unique_minutes)
    split_idx  = int(n * train_frac)
    train_min  = unique_minutes.iloc[:split_idx]
    hold_min   = unique_minutes.iloc[split_idx:]
    train_df   = df[df["Minute"].isin(train_min)]
    holdout_df = df[df["Minute"].isin(hold_min)]
    return train_df, holdout_df
