"""Master pipeline. Run with: python -m v9_final.run_all --train-dir <dir> [options]"""
import argparse
import os
import sys

# Windows terminals may default to cp1252; force UTF-8 for Unicode print output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from .backtest import load_lob, precompute_context, run_strategy, project_metric, chronological_split
from .strategies import TWAP, STRATEGY_CLASSES
from .meta import microstructure_fingerprint, tune_strategy_on_train, autonomous_aapl_selection

KNOWN_STOCKS = ["AMZN", "GOOG", "INTC", "MSFT"]


def _banner(stock):
    print(f"\n{'='*60}")
    print(f"  {stock}")
    print(f"{'='*60}")


def run_stock(stock, train_path, test_path=None, verbose=True):
    if verbose:
        _banner(stock)

    df = load_lob(train_path)
    n_rows = len(df)
    n_min  = df["Minute"].nunique()
    if verbose:
        print(f"  Loaded {n_rows:,} rows | {n_min} minutes")

    inner_train, inner_holdout = chronological_split(df, train_frac=0.7)
    if verbose:
        print(f"  Train: {inner_train['Minute'].nunique()} min  |  "
              f"Holdout: {inner_holdout['Minute'].nunique()} min")

    if verbose:
        print("  Tuning on train split ...")
    tune = tune_strategy_on_train(inner_train, verbose=verbose)
    best = tune

    if verbose:
        print(f"\n  Best: {best['strategy_name']}  {best['params']}"
              f"  ->  train {best['train_improvement_pct']:.2f}%")

    # Holdout evaluation
    ctx_h  = precompute_context(inner_holdout)
    twap   = TWAP()
    tw_h   = run_strategy(inner_holdout, twap, ctx_h)
    cls    = STRATEGY_CLASSES[best["strategy_name"]]
    strat  = cls(**best["params"])
    al_h   = run_strategy(inner_holdout, strat, ctx_h)
    met_h  = project_metric(tw_h, al_h)
    holdout_metric = met_h["improvement_pct"]
    retention = (holdout_metric / best["train_improvement_pct"]
                 if best["train_improvement_pct"] != 0 else 0.0)

    if verbose:
        print(f"  Holdout: {holdout_metric:.2f}%  |  retention: {retention:.2%}")
        if retention < 0.5:
            print("  [!] retention < 50% -- possible overfitting")

    # Full-train evaluation (for report)
    ctx_f  = precompute_context(df)
    tw_f   = run_strategy(df, twap,  ctx_f)
    al_f   = run_strategy(df, strat, ctx_f)
    met_f  = project_metric(tw_f, al_f)

    fp = microstructure_fingerprint(df)

    # Test set
    test_metric = None
    if test_path and os.path.exists(test_path):
        if verbose:
            print("  Evaluating on test set …")
        df_t  = load_lob(test_path)
        ctx_t = precompute_context(df_t)
        tw_t  = run_strategy(df_t, twap,  ctx_t)
        al_t  = run_strategy(df_t, strat, ctx_t)
        test_metric = project_metric(tw_t, al_t)
        if verbose:
            print(f"  Test:    {test_metric['improvement_pct']:.2f}%")

    return {
        "stock":          stock,
        "best":           best,
        "fingerprint":    fp,
        "train_metric":   best["train_improvement_pct"],
        "holdout_metric": holdout_metric,
        "retention":      retention,
        "test_metric":    test_metric,
        "full_metric":    met_f,
        "twap_full":      tw_f,
        "algo_full":      al_f,
        "twap_hold":      tw_h,
        "algo_hold":      al_h,
        "metric_hold":    met_h,
    }


def run_aapl(train_path, known_stocks, test_path=None, verbose=True):
    if verbose:
        _banner("AAPL  (autonomous selection)")

    df = load_lob(train_path)
    if verbose:
        print(f"  Loaded {len(df):,} rows | {df['Minute'].nunique()} minutes")

    inner_train, inner_holdout = chronological_split(df, train_frac=0.7)
    if verbose:
        print(f"  Train: {inner_train['Minute'].nunique()} min  |  "
              f"Holdout: {inner_holdout['Minute'].nunique()} min")

    selection = autonomous_aapl_selection(inner_train, known_stocks, verbose=verbose)

    best = {
        "strategy_name":        selection["strategy_name"],
        "params":               selection["params"],
        "train_improvement_pct": selection["train_improvement_pct"],
        "all_scores":           selection.get("all_scores"),
    }

    # Holdout
    ctx_h = precompute_context(inner_holdout)
    twap  = TWAP()
    tw_h  = run_strategy(inner_holdout, twap, ctx_h)
    cls   = STRATEGY_CLASSES[best["strategy_name"]]
    strat = cls(**best["params"])
    al_h  = run_strategy(inner_holdout, strat, ctx_h)
    met_h = project_metric(tw_h, al_h)
    holdout_metric = met_h["improvement_pct"]
    retention = (holdout_metric / best["train_improvement_pct"]
                 if best["train_improvement_pct"] != 0 else 0.0)

    if verbose:
        print(f"\n  Holdout: {holdout_metric:.2f}%  |  retention: {retention:.2%}")
        if retention < 0.5:
            print("  [!] retention < 50% -- possible overfitting")

    # Full-train
    ctx_f = precompute_context(df)
    tw_f  = run_strategy(df, twap,  ctx_f)
    al_f  = run_strategy(df, strat, ctx_f)
    met_f = project_metric(tw_f, al_f)

    fp = microstructure_fingerprint(df)

    test_metric = None
    if test_path and os.path.exists(test_path):
        df_t  = load_lob(test_path)
        ctx_t = precompute_context(df_t)
        tw_t  = run_strategy(df_t, twap,  ctx_t)
        al_t  = run_strategy(df_t, strat, ctx_t)
        test_metric = project_metric(tw_t, al_t)
        if verbose:
            print(f"  Test:    {test_metric['improvement_pct']:.2f}%")

    return {
        "stock":          "AAPL",
        "best":           best,
        "fingerprint":    fp,
        "train_metric":   best["train_improvement_pct"],
        "holdout_metric": holdout_metric,
        "retention":      retention,
        "test_metric":    test_metric,
        "full_metric":    met_f,
        "twap_full":      tw_f,
        "algo_full":      al_f,
        "twap_hold":      tw_h,
        "algo_hold":      al_h,
        "metric_hold":    met_h,
        "selection":      selection,
    }


def run_full_pipeline(train_dir, test_dir=None):
    pipeline     = {}
    known_stocks = {}

    for stock in KNOWN_STOCKS:
        train_path = os.path.join(train_dir, f"{stock}_5levels_train.csv")
        if not os.path.exists(train_path):
            print(f"  Skipping {stock}: {train_path} not found")
            continue
        test_path = (os.path.join(test_dir, f"{stock}_5levels_test.csv")
                     if test_dir else None)
        result = run_stock(stock, train_path, test_path, verbose=True)
        pipeline[stock]     = result
        known_stocks[stock] = result

    aapl_train = os.path.join(train_dir, "AAPL_5levels_train.csv")
    if os.path.exists(aapl_train):
        aapl_test = (os.path.join(test_dir, "AAPL_5levels_test.csv")
                     if test_dir else None)
        result = run_aapl(aapl_train, known_stocks, aapl_test, verbose=True)
        pipeline["AAPL"] = result
    else:
        print(f"\nAAPL training data not found at {aapl_train} — skipping.")

    return pipeline


def build_results_table(pipeline):
    rows = []
    for stock, data in pipeline.items():
        best  = data["best"]
        met_f = data.get("full_metric") or {}
        tm    = data.get("test_metric")

        rows.append({
            "Stock":             stock,
            "Strategy":          best["strategy_name"],
            "Params":            str(best["params"]),
            "Train Improv (%)":  round(data["train_metric"],   2),
            "Holdout Improv (%)": round(data["holdout_metric"], 2),
            "Retention Ratio":   round(data["retention"],       3),
            "Test Improv (%)":   round(tm["improvement_pct"],  2) if tm else None,
            "Train Buy α (bps)": round(met_f.get("buy_improv_bps",  0), 2),
            "Train Sell α (bps)": round(met_f.get("sell_improv_bps", 0), 2),
            "TWAP Cost":         round(met_f.get("twap_cost", 0), 4),
            "Algo Cost":         round(met_f.get("algo_cost", 0), 4),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="V9 Final Pipeline")
    parser.add_argument("--train-dir", required=True,
                        help="Directory containing training CSVs")
    parser.add_argument("--test-dir",  default=None,
                        help="Directory containing test CSVs (optional)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data into --train-dir first")
    args = parser.parse_args()

    if args.synthetic:
        print("Generating synthetic data …")
        from .make_synthetic_data import generate_all
        os.makedirs(args.train_dir, exist_ok=True)
        generate_all(args.train_dir)
        print("Synthetic data ready.\n")

    pipeline = run_full_pipeline(args.train_dir, args.test_dir)
    table    = build_results_table(pipeline)

    print("\n" + "=" * 80)
    print("FINAL RESULTS TABLE")
    print("=" * 80)
    print(table.to_string(index=False))

    out_csv = os.path.join(args.train_dir, "v9_results_table.csv")
    table.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    main()
