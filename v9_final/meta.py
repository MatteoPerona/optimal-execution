import numpy as np
import pandas as pd

from .backtest import precompute_context, run_strategy, project_metric, chronological_split
from .strategies import TWAP, STRATEGY_CLASSES, PARAM_GRIDS


def microstructure_fingerprint(df):
    mid    = df["MidPrice"]
    spread = df["Spread"]

    avg_mid    = float(mid.mean())
    spread_bps = float((spread / mid.replace(0, np.nan)).median() * 1e4)

    tick_freq  = float(df.groupby("Minute").size().mean())

    depth      = df["BidSize_1"] + df["AskSize_1"]
    sp_bps_tk  = (spread / mid.replace(0, np.nan) * 1e4).replace(0, np.nan)
    depth_per_bps = float((depth / sp_bps_tk).median())

    min_mids   = df.groupby("Minute")["MidPrice"].last()
    ret_bps    = min_mids.pct_change().dropna() * 1e4
    vol_bps    = float(ret_bps.std())

    return {
        "avg_mid":       avg_mid,
        "spread_bps":    spread_bps,
        "tick_freq":     tick_freq,
        "depth_per_bps": depth_per_bps,
        "vol_bps":       vol_bps,
    }


def fingerprint_distance(fp1, fp2):
    keys = ["spread_bps", "tick_freq", "depth_per_bps", "vol_bps"]
    dist = 0.0
    for k in keys:
        v1, v2 = fp1[k], fp2[k]
        if v1 > 0 and v2 > 0:
            dist += (np.log(v1) - np.log(v2)) ** 2
        else:
            dist += (v1 - v2) ** 2
    return float(np.sqrt(dist))


def tune_strategy_on_train(train_df, verbose=False):
    ctx        = precompute_context(train_df)
    twap       = TWAP()
    twap_res   = run_strategy(train_df, twap, ctx)

    best_score = -np.inf
    best_name  = None
    best_params = None
    scores     = []

    for name, cls in STRATEGY_CLASSES.items():
        if name == "TWAP":
            continue
        param_list = PARAM_GRIDS.get(name, [{}])
        for params in param_list:
            try:
                strat  = cls(**params)
                result = run_strategy(train_df, strat, ctx)
                metric = project_metric(twap_res, result)
                score  = metric["improvement_pct"]
                scores.append({
                    "strategy_name":        name,
                    "params":               str(params),
                    "train_improvement_pct": score,
                })
                if verbose:
                    print(f"    {name} {params}: {score:.2f}%")
                if score > best_score:
                    best_score  = score
                    best_name   = name
                    best_params = params
            except Exception as e:
                if verbose:
                    print(f"    {name} {params}: ERROR — {e}")

    # Fallback: if everything failed, use TWAP-equivalent
    if best_name is None:
        best_name   = "SpreadQuantile"
        best_params = {}
        best_score  = 0.0

    return {
        "strategy_name":        best_name,
        "params":               best_params,
        "train_improvement_pct": best_score,
        "all_scores":           pd.DataFrame(scores),
    }


def autonomous_aapl_selection(aapl_train_df, known_stocks, verbose=True):
    log = []

    aapl_fp = microstructure_fingerprint(aapl_train_df)
    if verbose:
        print(f"\n  AAPL fingerprint: {aapl_fp}")
    log.append(f"AAPL fingerprint: {aapl_fp}")

    distances = {}
    for stock, info in known_stocks.items():
        d = fingerprint_distance(aapl_fp, info["fingerprint"])
        distances[stock] = d
        if verbose:
            print(f"    Distance to {stock}: {d:.4f}")
        log.append(f"Distance to {stock}: {d:.4f}")

    nearest = min(distances, key=distances.get)
    if verbose:
        print(f"  Nearest neighbor: {nearest}")
    log.append(f"Nearest neighbor: {nearest}")

    # Route A — transfer nearest neighbor's winning strategy
    ra_name   = known_stocks[nearest]["best"]["strategy_name"]
    ra_params = known_stocks[nearest]["best"]["params"]

    ctx_a      = precompute_context(aapl_train_df)
    twap       = TWAP()
    twap_res   = run_strategy(aapl_train_df, twap, ctx_a)
    cls_a      = STRATEGY_CLASSES[ra_name]
    res_a      = run_strategy(aapl_train_df, cls_a(**ra_params), ctx_a)
    route_a_score = project_metric(twap_res, res_a)["improvement_pct"]
    if verbose:
        print(f"  Route A ({nearest}'s {ra_name}): {route_a_score:.2f}%")
    log.append(f"Route A ({nearest}'s {ra_name}): {route_a_score:.2f}%")

    # Route B — direct grid search on AAPL train
    if verbose:
        print("  Route B: direct grid-search on AAPL train …")
    tune       = tune_strategy_on_train(aapl_train_df, verbose=verbose)
    route_b_score   = tune["train_improvement_pct"]
    rb_name    = tune["strategy_name"]
    rb_params  = tune["params"]
    if verbose:
        print(f"  Route B ({rb_name}): {route_b_score:.2f}%")
    log.append(f"Route B ({rb_name}): {route_b_score:.2f}%")

    if route_a_score >= route_b_score:
        decision      = "Route A (nearest-neighbor transfer)"
        final_name    = ra_name
        final_params  = ra_params
        final_score   = route_a_score
    else:
        decision      = "Route B (direct grid-search)"
        final_name    = rb_name
        final_params  = rb_params
        final_score   = route_b_score

    if verbose:
        print(f"  Decision: {decision} -> {final_name} {final_params}")
    log.append(f"Decision: {decision} -> {final_name} {final_params}")

    return {
        "decision":             decision,
        "strategy_name":        final_name,
        "params":               final_params,
        "train_improvement_pct": final_score,
        "fingerprint":          aapl_fp,
        "distances":            distances,
        "nearest":              nearest,
        "route_a_score":        route_a_score,
        "route_b_score":        route_b_score,
        "all_scores":           tune["all_scores"],
        "log":                  log,
    }
