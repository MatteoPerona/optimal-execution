"""
Comprehensive presentation plots covering the full project arc:
Milestone 1 (V2/V5/V6/V7/V8) through V9 Final.

Generates 9 figures to <out_dir>/ and prints professor-format
describe() tables to stdout.

Usage:
    from v9_final.make_presentation import run_presentation
    run_presentation(
        train_dir  = "./1. Project Files",
        project_dir= "./1. Project Files",
        out_dir    = "./1. Project Files/v9_presentation"
    )

Or from the CLI:
    python -m v9_final.make_presentation
"""
import os
import sys
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Colour scheme ─────────────────────────────────────────────────────────────
NAVY   = "#002060"
BLUE   = "#1565C0"
RED    = "#C62828"
GREEN  = "#2E7D32"
PURPLE = "#6A1B9A"
AMBER  = "#E65100"
TEAL   = "#00695C"
GREY   = "#546E7A"
LIGHT  = "#ECEFF1"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Historical milestone data
# ═══════════════════════════════════════════════════════════════════════════════

def _v8_hardcoded():
    """V8 OFI-Adaptive OOS results read from the saved chart image."""
    return pd.DataFrame([
        {"Stock": "AMZN",      "Buy_bps": -0.447, "Sell_bps": -0.948, "Overall_bps": -0.697},
        {"Stock": "GOOG",      "Buy_bps":  0.232, "Sell_bps": -0.839, "Overall_bps": -0.303},
        {"Stock": "INTC",      "Buy_bps": -0.409, "Sell_bps": -1.757, "Overall_bps": -1.083},
        {"Stock": "MSFT",      "Buy_bps": -0.639, "Sell_bps": -1.556, "Overall_bps": -1.097},
        {"Stock": "PORTFOLIO", "Buy_bps":  0.003, "Sell_bps": -0.922, "Overall_bps": -0.460},
    ])


def load_milestone_data(project_dir):
    """
    Load all historical milestone CSVs.
    Returns a dict keyed by version label, each holding a DataFrame
    with columns: Stock, Buy_bps, Sell_bps, Overall_bps.
    """
    def _read(name, b_col, s_col, o_col, rename_stock="Stock",
              rename_regime=None, filter_portfolio=True):
        path = os.path.join(project_dir, name)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        if rename_regime and rename_regime in df.columns:
            df = df.rename(columns={rename_regime: "Regime"})
        out = df[[rename_stock, b_col, s_col, o_col]].copy()
        out.columns = ["Stock", "Buy_bps", "Sell_bps", "Overall_bps"]
        return out

    versions = {}

    v5_is = _read("V5_Table_InSample.csv",
                  "Buy Improvement (bps)", "Sell Improvement (bps)", "Overall (bps)")
    if v5_is is not None:
        versions["V5 In-Sample"] = v5_is

    v5_oos = _read("V5_Table_OOS.csv",
                   "Buy Improvement (bps)", "Sell Improvement (bps)", "Overall (bps)")
    if v5_oos is not None:
        versions["V5 OOS"] = v5_oos

    v6_is = _read("V6_Table_In-Sample.csv", "Buy α (bps)", "Sell α (bps)", "Overall (bps)")
    if v6_is is not None:
        versions["V6 In-Sample"] = v6_is

    v7_dev = _read("V7_Table_Dev_Performance.csv", "Buy α (bps)", "Sell α (bps)", "Overall (bps)")
    if v7_dev is not None:
        versions["V7 Dev"] = v7_dev

    v7_oos = _read("V7_Table_OOS_Performance.csv", "Buy α (bps)", "Sell α (bps)", "Overall (bps)")
    if v7_oos is not None:
        versions["V7 OOS"] = v7_oos

    versions["V8 OOS"] = _v8_hardcoded()

    return versions


def _portfolio_row(df, label):
    """Extract or compute the PORTFOLIO / ALL row."""
    mask = df["Stock"].isin(["PORTFOLIO", "ALL"])
    if mask.any():
        return df[mask].iloc[0]
    # Compute simple mean of individual stocks
    stocks = df[~df["Stock"].isin(["PORTFOLIO", "ALL"])]
    return pd.Series({
        "Stock":       "PORTFOLIO",
        "Buy_bps":     stocks["Buy_bps"].mean(),
        "Sell_bps":    stocks["Sell_bps"].mean(),
        "Overall_bps": stocks["Overall_bps"].mean(),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — V9 live results
# ═══════════════════════════════════════════════════════════════════════════════

def run_v9(train_dir, verbose=False):
    """Run v9 pipeline and return (pipeline, table)."""
    from .run_all import run_full_pipeline, build_results_table
    pipe  = run_full_pipeline(train_dir)
    table = build_results_table(pipe)
    return pipe, table


def extract_v9_alpha_summary(pipe, table):
    """
    Returns two DataFrames:
      train_df  — full-train Buy/Sell/Overall alpha (bps) per stock
      hold_df   — holdout  Buy/Sell/Overall alpha (bps) per stock
    """
    stocks = ["AMZN", "GOOG", "INTC", "MSFT", "AAPL"]
    train_rows, hold_rows = [], []

    for stock in stocks:
        if stock not in pipe:
            continue
        d = pipe[stock]
        mf = d.get("full_metric", {})
        mh = d.get("metric_hold", {})

        train_rows.append({
            "Stock":       stock,
            "Buy_bps":     mf.get("buy_improv_bps",  0),
            "Sell_bps":    mf.get("sell_improv_bps", 0),
            "Overall_bps": (mf.get("buy_improv_bps", 0) + mf.get("sell_improv_bps", 0)) / 2,
        })
        hold_rows.append({
            "Stock":       stock,
            "Buy_bps":     mh.get("buy_improv_bps",  0),
            "Sell_bps":    mh.get("sell_improv_bps", 0),
            "Overall_bps": (mh.get("buy_improv_bps", 0) + mh.get("sell_improv_bps", 0)) / 2,
        })

    train_df = pd.DataFrame(train_rows)
    hold_df  = pd.DataFrame(hold_rows)

    for df in (train_df, hold_df):
        vals = df[df["Stock"] != "AAPL"]
        df.loc[len(df)] = {
            "Stock":       "PORTFOLIO",
            "Buy_bps":     vals["Buy_bps"].mean(),
            "Sell_bps":    vals["Sell_bps"].mean(),
            "Overall_bps": vals["Overall_bps"].mean(),
        }

    return train_df, hold_df


def compute_per_minute_stats(pipe):
    """
    For each stock in the pipeline, compute per-minute improvement series.
    Returns dict: stock -> {buy_impr_$, sell_impr_$, buy_impr_bps, sell_impr_bps, mid_approx}
    """
    results = {}
    for stock, d in pipe.items():
        tw = d.get("twap_full", {})
        al = d.get("algo_full", {})
        if not tw or not al:
            continue
        bp_tw = tw["buy_prices"]
        bp_al = al["buy_prices"]
        sp_tw = tw["sell_prices"]
        sp_al = al["sell_prices"]

        mid = (bp_tw + sp_tw) / 2.0

        buy_impr_dol  = bp_tw - bp_al           # positive = bought cheaper
        sell_impr_dol = sp_al - sp_tw           # positive = sold dearer
        buy_impr_bps  = buy_impr_dol  / mid * 1e4
        sell_impr_bps = sell_impr_dol / mid * 1e4

        results[stock] = {
            "buy_impr_dol":  buy_impr_dol,
            "sell_impr_dol": sell_impr_dol,
            "buy_impr_bps":  buy_impr_bps,
            "sell_impr_bps": sell_impr_bps,
            "mid":           mid,
            "avg_twap_buy":  float(bp_tw.mean()),
            "avg_algo_buy":  float(bp_al.mean()),
            "avg_twap_sell": float(sp_tw.mean()),
            "avg_algo_sell": float(sp_al.mean()),
            "avg_twap_spread": float((bp_tw - sp_tw).mean()),
            "avg_algo_spread": float((bp_al - sp_al).mean()),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Console tables (professor-format)
# ═══════════════════════════════════════════════════════════════════════════════

def print_describe_tables(per_min, stock):
    """Print BUY/SELL IMPROVEMENT describe() tables in professor format."""
    if stock not in per_min:
        return
    d = per_min[stock]
    sep = "=" * 30
    print(f"\n{sep}")
    print(f"{stock} — BUY IMPROVEMENT ($)")
    print(sep)
    print(d["buy_impr_dol"].describe().to_string())
    print(f"\n{sep}")
    print(f"{stock} — SELL IMPROVEMENT ($)")
    print(sep)
    print(d["sell_impr_dol"].describe().to_string())
    print(f"\n{sep}")
    print(f"{stock} — TWAP vs ALGO COMPARISON")
    print(sep)
    rows = [
        ("Average Buy Price",    d["avg_twap_buy"],    d["avg_algo_buy"],
         d["avg_twap_buy"]  - d["avg_algo_buy"]),
        ("Average Sell Price",   d["avg_twap_sell"],   d["avg_algo_sell"],
         d["avg_algo_sell"] - d["avg_twap_sell"]),
        ("Average BUY-SELL Spread", d["avg_twap_spread"], d["avg_algo_spread"],
         d["avg_twap_spread"] - d["avg_algo_spread"]),
    ]
    cmp = pd.DataFrame(rows, columns=["Metric", "TWAP", "Algo", "Improvement"])
    print(cmp.to_string(index=False, float_format="{:.5f}".format))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Plots
# ═══════════════════════════════════════════════════════════════════════════════

def _savefig(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


# ── Plot 1: Portfolio α Journey ───────────────────────────────────────────────

def plot_alpha_journey(milestones, v9_train_df, v9_hold_df, out):
    """
    Portfolio Overall α (bps) from V5 In-Sample through V9 Holdout.
    Green bar = positive (algo beats TWAP), Red = negative.
    """
    labels, vals, colors, hatches = [], [], [], []

    version_order = ["V5 In-Sample", "V5 OOS",
                     "V6 In-Sample",
                     "V7 Dev", "V7 OOS",
                     "V8 OOS",
                     "V9 Train", "V9 Holdout"]

    milestone_alpha = {}
    for ver, df in milestones.items():
        row = _portfolio_row(df, ver)
        milestone_alpha[ver] = float(row["Overall_bps"])

    v9_row_tr = v9_train_df[v9_train_df["Stock"] == "PORTFOLIO"]
    v9_row_ho = v9_hold_df[v9_hold_df["Stock"] == "PORTFOLIO"]
    milestone_alpha["V9 Train"]   = float(v9_row_tr["Overall_bps"].iloc[0]) if len(v9_row_tr) else 0
    milestone_alpha["V9 Holdout"] = float(v9_row_ho["Overall_bps"].iloc[0]) if len(v9_row_ho) else 0

    for ver in version_order:
        if ver not in milestone_alpha:
            continue
        v = milestone_alpha[ver]
        labels.append(ver)
        vals.append(v)
        colors.append(GREEN if v >= 0 else RED)
        hatches.append("///" if "OOS" in ver or "Holdout" in ver else "")

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, hatch=hatches, edgecolor="white", linewidth=0.8, width=0.6)

    for bar, v in zip(bars, vals):
        ypos = bar.get_height() + 0.03 if v >= 0 else bar.get_height() - 0.10
        ax.text(bar.get_x() + bar.get_width() / 2, ypos, f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=9, fontweight="bold",
                color=GREEN if v >= 0 else RED)

    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(5.5, color=GREY, linewidth=1, linestyle="--", alpha=0.6)
    ax.text(5.7, ax.get_ylim()[0] * 0.7, "V9 Walk-Forward\n(our answer to M1 critique)",
            fontsize=8, color=GREY, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Portfolio Overall α vs TWAP (bps)", fontsize=11)
    ax.set_title("Algorithm Alpha — Full Project Arc\n"
                 "Hatched bars = out-of-sample / holdout periods",
                 fontsize=13, fontweight="bold")

    legend_els = [
        mpatches.Patch(facecolor=GREEN, label="Alpha > 0 (beats TWAP)"),
        mpatches.Patch(facecolor=RED,   label="Alpha < 0 (loses to TWAP)"),
        mpatches.Patch(facecolor="white", hatch="///", edgecolor=GREY, label="OOS / Holdout period"),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc="lower right")
    _savefig(fig, out)


# ── Plot 2: Buy vs Sell α decomposition ──────────────────────────────────────

def plot_buysell_decomposition(milestones, v9_train_df, out):
    """
    Buy α and Sell α side-by-side per version for the portfolio.
    Reveals that old strategies hurt sell side; V9 fixes both.
    """
    vers, buys, sells = [], [], []

    for ver, df in milestones.items():
        row = _portfolio_row(df, ver)
        vers.append(ver)
        buys.append(float(row["Buy_bps"]))
        sells.append(float(row["Sell_bps"]))

    row = v9_train_df[v9_train_df["Stock"] == "PORTFOLIO"]
    if len(row):
        vers.append("V9 Full-Train")
        buys.append(float(row["Buy_bps"].iloc[0]))
        sells.append(float(row["Sell_bps"].iloc[0]))

    x    = np.arange(len(vers))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - w / 2, buys,  w, label="Buy α (bps)",  color=BLUE,  edgecolor="white")
    b2 = ax.bar(x + w / 2, sells, w, label="Sell α (bps)", color=GREEN, edgecolor="white")

    for bar, v in list(zip(b1, buys)) + list(zip(b2, sells)):
        yoff = 0.04 if v >= 0 else -0.10
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + yoff,
                f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=7.5, color="black")

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(vers, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Portfolio Alpha vs TWAP (bps)", fontsize=11)
    ax.set_title("Buy vs Sell Alpha Decomposition — Portfolio Level\n"
                 "Key story: earlier versions hurt sell side; V9 achieves positive alpha on both legs",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.annotate("V9 first version\nwith +ve sell α",
                xy=(len(vers) - 1, sells[-1]),
                xytext=(len(vers) - 2.2, sells[-1] + 1.0),
                fontsize=8, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
    _savefig(fig, out)


# ── Plot 3: Per-stock alpha across versions ───────────────────────────────────

def plot_perstock_progression(milestones, v9_train_df, out):
    """
    Overall α per stock across every version — shows which stocks were
    consistently hard/easy to improve.
    """
    stocks = ["AMZN", "GOOG", "INTC", "MSFT"]
    version_labels = list(milestones.keys()) + ["V9 Full-Train"]

    data = {s: [] for s in stocks}
    for ver, df in milestones.items():
        for s in stocks:
            row = df[df["Stock"] == s]
            data[s].append(float(row["Overall_bps"].iloc[0]) if len(row) else np.nan)

    v9_row = v9_train_df[v9_train_df["Stock"].isin(stocks)]
    for s in stocks:
        r = v9_row[v9_row["Stock"] == s]
        data[s].append(float(r["Overall_bps"].iloc[0]) if len(r) else np.nan)

    colors_s = [NAVY, BLUE, RED, GREEN]
    markers   = ["o", "s", "^", "D"]
    x = np.arange(len(version_labels))

    fig, ax = plt.subplots(figsize=(13, 6))
    for (s, c, m) in zip(stocks, colors_s, markers):
        vals = data[s]
        ax.plot(x, vals, marker=m, color=c, linewidth=2, markersize=7, label=s)
        ax.annotate(f"{vals[-1]:+.2f} bps",
                    xy=(x[-1], vals[-1]),
                    xytext=(x[-1] + 0.1, vals[-1]),
                    fontsize=8, color=c)

    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(version_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Overall α vs TWAP (bps)", fontsize=11)
    ax.set_title("Per-Stock Algorithm Alpha — Version Progression\n"
                 "V9 achieves positive alpha on all four stocks simultaneously",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower left")
    _savefig(fig, out)


# ── Plot 4: V9 Walk-Forward Validation ───────────────────────────────────────

def plot_v9_walkforward(pipe, table, out):
    """
    Train vs Holdout improvement % per stock with retention ratio labels.
    Directly addresses the professors' Milestone 1 critique.
    """
    stocks = [s for s in ["AMZN", "GOOG", "INTC", "MSFT", "AAPL"] if s in pipe]
    train  = [pipe[s]["train_metric"]   for s in stocks]
    hold   = [pipe[s]["holdout_metric"] for s in stocks]
    ret    = [pipe[s]["retention"]      for s in stocks]

    x = np.arange(len(stocks))
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(11, 6))
    b1 = ax1.bar(x - w / 2, train, w, label="Train improvement (%)",   color=NAVY, edgecolor="white")
    b2 = ax1.bar(x + w / 2, hold,  w, label="Holdout improvement (%)", color=BLUE, edgecolor="white",
                 hatch="///")

    # Retention ratio labels above each stock
    for i, (t, h, r) in enumerate(zip(train, hold, ret)):
        color  = GREEN if r > 0.7 else (AMBER if r > 0.5 else RED)
        symbol = "✓" if r > 0.7 else ("~" if r > 0.5 else "!")
        ax1.text(i, max(t, h) + 2, f"Retention: {r:.0%} {symbol}",
                 ha="center", fontsize=8.5, fontweight="bold", color=color)

    ax1.set_xticks(x)
    ax1.set_xticklabels(stocks, fontsize=11)
    ax1.set_ylabel("Improvement vs TWAP (%)", fontsize=11)
    ax1.set_title("V9 Walk-Forward Validation — Train vs Holdout\n"
                  "70/30 Chronological Split  |  Retention = Holdout / Train",
                  fontsize=12, fontweight="bold")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.legend(fontsize=10, loc="upper left")

    # Retention threshold guide lines (on secondary axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Retention Ratio", fontsize=10, color=GREY)
    ax2.axhline(0.70, color=AMBER, linewidth=1.2, linestyle=":", alpha=0.7, label="70% threshold")
    ax2.axhline(0.50, color=RED,   linewidth=1.2, linestyle=":", alpha=0.7, label="50% threshold")
    ax2.plot(x, ret, "o--", color=PURPLE, linewidth=1.5, markersize=8, label="Retention ratio", zorder=5)
    ax2.set_ylim(0, 1.5)
    ax2.tick_params(axis="y", labelcolor=GREY)
    ax2.legend(fontsize=8.5, loc="upper right")

    _savefig(fig, out)


# ── Plot 5: Retention ratio bar ───────────────────────────────────────────────

def plot_retention_flags(pipe, out):
    """
    Standalone retention ratio bar chart with colour-coded risk zones.
    """
    stocks = [s for s in ["AMZN", "GOOG", "INTC", "MSFT", "AAPL"] if s in pipe]
    ret    = [pipe[s]["retention"] for s in stocks]
    colors = [GREEN if r > 0.7 else (AMBER if r > 0.5 else RED) for r in ret]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(stocks, [r * 100 for r in ret], color=colors, edgecolor="white", width=0.5)

    for bar, r in zip(bars, ret):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{r:.1%}", ha="center", fontsize=11, fontweight="bold")

    ax.axhline(70, color=AMBER, linewidth=1.5, linestyle="--", label="70% — robust threshold")
    ax.axhline(50, color=RED,   linewidth=1.5, linestyle="--", label="50% — overfitting concern")
    ax.set_ylim(0, 130)
    ax.set_ylabel("Retention Ratio (%)", fontsize=11)
    ax.set_title("Walk-Forward Retention Ratio by Stock\n"
                 "Green > 70%  |  Amber 50–70%  |  Red < 50%",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # Background zones
    ax.axhspan(70, 130, alpha=0.04, color=GREEN)
    ax.axhspan(50, 70,  alpha=0.05, color=AMBER)
    ax.axhspan(0,  50,  alpha=0.04, color=RED)

    _savefig(fig, out)


# ── Plot 6: Per-minute improvement distributions ──────────────────────────────

def plot_improvement_distributions(per_min, out):
    """
    Box plots of per-minute buy and sell improvement (bps) for each stock.
    Shows that improvements are asymmetric: 0 when no better tick found,
    positive tail when Ensemble fires successfully.
    """
    stocks = [s for s in ["AMZN", "GOOG", "INTC", "MSFT"] if s in per_min]
    if not stocks:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    buy_data  = [per_min[s]["buy_impr_bps"].values  for s in stocks]
    sell_data = [per_min[s]["sell_impr_bps"].values for s in stocks]

    for ax, data, side, color, label in [
        (axes[0], buy_data,  "BUY",  BLUE,  "Buy α (bps)"),
        (axes[1], sell_data, "SELL", GREEN, "Sell α (bps)"),
    ]:
        bp = ax.boxplot(data, labels=stocks, patch_artist=True,
                        medianprops={"color": "white", "linewidth": 2},
                        whiskerprops={"linewidth": 1.2},
                        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4, "color": GREY})
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(f"{side} Improvement Distribution (bps)\nper minute vs TWAP",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel(label, fontsize=10)
        ax.set_xlabel("Stock", fontsize=10)

        # Add mean annotations
        for i, (d, s) in enumerate(zip(data, stocks)):
            mean_v = float(np.mean(d))
            ax.text(i + 1, ax.get_ylim()[1] * 0.93, f"μ={mean_v:.2f}",
                    ha="center", fontsize=8, color="black")

    fig.suptitle("V9 Ensemble — Per-Minute Execution Alpha Distribution\n"
                 "Zero mass = minutes falling back to TWAP;  positive tail = successful timing",
                 fontsize=12, fontweight="bold")
    _savefig(fig, out)


# ── Plot 7: Descriptive stats tables ─────────────────────────────────────────

def plot_describe_tables(per_min, out):
    """
    Render the professor-format describe() tables as a matplotlib figure.
    One column per stock, BUY on top / SELL on bottom.
    """
    stocks = [s for s in ["AMZN", "GOOG", "INTC", "MSFT"] if s in per_min]
    if not stocks:
        return

    stat_keys = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

    fig, axes = plt.subplots(2, len(stocks), figsize=(4.5 * len(stocks), 9))
    if len(stocks) == 1:
        axes = axes.reshape(2, 1)

    for col, stock in enumerate(stocks):
        d = per_min[stock]
        for row, (series, side) in enumerate([
            (d["buy_impr_dol"],  "BUY"),
            (d["sell_impr_dol"], "SELL"),
        ]):
            ax = axes[row, col]
            ax.axis("off")

            desc = series.describe()
            cell_text = [[f"{desc[k]:>12.6f}"] for k in stat_keys]
            tbl = ax.table(
                cellText=cell_text,
                rowLabels=stat_keys,
                colLabels=[f"{side} IMPROVEMENT ($)"],
                loc="center",
                cellLoc="right",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1.3, 1.6)

            # Header styling
            for (r, c), cell in tbl.get_celld().items():
                if r == 0:
                    cell.set_facecolor(NAVY if side == "BUY" else GREEN)
                    cell.set_text_props(color="white", fontweight="bold")
                elif r % 2 == 0:
                    cell.set_facecolor(LIGHT)
                cell.set_edgecolor("white")

            ax.set_title(f"{stock} — {side}", fontsize=11, fontweight="bold",
                         color=NAVY if side == "BUY" else GREEN, pad=4)

    fig.suptitle("V9 Per-Minute Execution Improvement — Descriptive Statistics\n"
                 "(improvement = TWAP price − algo price;  positive = algo beats TWAP)",
                 fontsize=12, fontweight="bold", y=1.01)
    _savefig(fig, out, tight=False)


# ── Plot 8: TWAP vs Algo comparison table ─────────────────────────────────────

def plot_twap_comparison(per_min, out):
    """
    Formatted TWAP vs Algo comparison table per stock.
    Matches the professor-requested format:
      Metric | TWAP | Algo | Improvement ($) | Improvement (bps)
    """
    stocks = [s for s in ["AMZN", "GOOG", "INTC", "MSFT"] if s in per_min]
    if not stocks:
        return

    fig, axes = plt.subplots(len(stocks), 1, figsize=(12, 2.5 * len(stocks)))
    if len(stocks) == 1:
        axes = [axes]

    for ax, stock in zip(axes, stocks):
        ax.axis("off")
        d = per_min[stock]
        mid = (d["avg_twap_buy"] + d["avg_twap_sell"]) / 2

        buy_improv_dol  = d["avg_twap_buy"]    - d["avg_algo_buy"]
        sell_improv_dol = d["avg_algo_sell"]   - d["avg_twap_sell"]
        spr_improv_dol  = d["avg_twap_spread"] - d["avg_algo_spread"]

        buy_improv_bps  = buy_improv_dol  / mid * 1e4
        sell_improv_bps = sell_improv_dol / mid * 1e4
        spr_improv_bps  = spr_improv_dol  / mid * 1e4

        row_data = [
            ["Average Buy Price",
             f"${d['avg_twap_buy']:.5f}", f"${d['avg_algo_buy']:.5f}",
             f"${buy_improv_dol:+.5f}", f"{buy_improv_bps:+.2f} bps"],
            ["Average Sell Price",
             f"${d['avg_twap_sell']:.5f}", f"${d['avg_algo_sell']:.5f}",
             f"${sell_improv_dol:+.5f}", f"{sell_improv_bps:+.2f} bps"],
            ["Avg BUY-SELL Spread",
             f"${d['avg_twap_spread']:.5f}", f"${d['avg_algo_spread']:.5f}",
             f"${spr_improv_dol:+.5f}", f"{spr_improv_bps:+.2f} bps"],
        ]
        col_labels = ["Metric", "TWAP", "Algo (V9 Ensemble)", "Improvement ($)", "Improvement (bps)"]

        tbl = ax.table(cellText=row_data, colLabels=col_labels,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.8)

        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("white")
            if r == 0:
                cell.set_facecolor(NAVY)
                cell.set_text_props(color="white", fontweight="bold")
            elif c == 3:  # improvement $ col
                val = row_data[r - 1][3].replace("$", "").replace(" bps", "")
                try:
                    cell.set_facecolor(GREEN if float(val) > 0 else RED)
                    cell.set_text_props(color="white", fontweight="bold")
                except ValueError:
                    pass
            elif c == 4:  # improvement bps col
                val = row_data[r - 1][4].replace(" bps", "")
                try:
                    cell.set_facecolor(GREEN if float(val) > 0 else RED)
                    cell.set_text_props(color="white", fontweight="bold")
                except ValueError:
                    pass
            elif r % 2 == 0:
                cell.set_facecolor(LIGHT)

        ax.set_title(f"{stock} — TWAP vs V9 Ensemble",
                     fontsize=11, fontweight="bold", loc="left", pad=6)

    fig.suptitle("V9 Execution Quality — TWAP vs Algorithm Comparison",
                 fontsize=13, fontweight="bold", y=1.01)
    _savefig(fig, out, tight=False)


# ── Plot 9: V9 full Buy/Sell α per stock ─────────────────────────────────────

def plot_v9_alpha_per_stock(pipe, out):
    """
    Per-stock Buy α and Sell α (full-train, in bps) for V9,
    with holdout values overlaid as outlined bars.
    """
    stocks = [s for s in ["AMZN", "GOOG", "INTC", "MSFT"] if s in pipe]
    if not stocks:
        return

    buy_tr  = [pipe[s]["full_metric"].get("buy_improv_bps",  0) for s in stocks]
    sell_tr = [pipe[s]["full_metric"].get("sell_improv_bps", 0) for s in stocks]
    buy_ho  = [pipe[s]["metric_hold"].get("buy_improv_bps",  0) for s in stocks]
    sell_ho = [pipe[s]["metric_hold"].get("sell_improv_bps", 0) for s in stocks]

    x  = np.arange(len(stocks))
    w  = 0.20
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(x - 1.5*w, buy_tr,  w, color=BLUE,  label="Buy α — Full Train",    edgecolor="white")
    ax.bar(x - 0.5*w, sell_tr, w, color=GREEN, label="Sell α — Full Train",   edgecolor="white")
    ax.bar(x + 0.5*w, buy_ho,  w, color=BLUE,  label="Buy α — Holdout",
           edgecolor=NAVY, linewidth=1.5, fill=False, hatch="///")
    ax.bar(x + 1.5*w, sell_ho, w, color=GREEN, label="Sell α — Holdout",
           edgecolor=GREEN, linewidth=1.5, fill=False, hatch="///")

    for bars, vals in [(ax.patches[:len(stocks)],   buy_tr),
                       (ax.patches[len(stocks):2*len(stocks)], sell_tr)]:
        pass  # annotations handled below

    for i, (bt, st, bh, sh) in enumerate(zip(buy_tr, sell_tr, buy_ho, sell_ho)):
        for xpos, v in [(i - 1.5*w, bt), (i - 0.5*w, st),
                        (i + 0.5*w, bh), (i + 1.5*w, sh)]:
            ax.text(xpos + w/2, v + 0.05 if v >= 0 else v - 0.12,
                    f"{v:.2f}", ha="center", fontsize=7.5, fontweight="bold")

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(stocks, fontsize=11)
    ax.set_ylabel("Alpha vs TWAP (bps)", fontsize=11)
    ax.set_title("V9 Ensemble — Buy α and Sell α per Stock\n"
                 "Train (solid) vs Holdout (hatched) — both sides positive",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, ncol=2)
    _savefig(fig, out)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Master runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_presentation(train_dir, project_dir, out_dir, verbose=False):
    """
    Full presentation suite.
    train_dir   : directory with *_5levels_train.csv files
    project_dir : directory with historical milestone CSVs (backtest_v2_summary, V5…V7 tables)
    out_dir     : where to save the PNGs
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutput directory: {out_dir}\n")

    # 1. Load milestone history
    print("Loading milestone data ...")
    milestones = load_milestone_data(project_dir)
    print(f"  Loaded versions: {list(milestones.keys())}")

    # 2. Run V9 pipeline
    print("\nRunning V9 pipeline (this takes a moment) ...")
    pipe, table = run_v9(train_dir, verbose=verbose)
    print(f"  Stocks processed: {list(pipe.keys())}")

    # 3. Derive V9 summary data
    v9_train_df, v9_hold_df = extract_v9_alpha_summary(pipe, table)
    per_min                  = compute_per_minute_stats(pipe)

    # 4. Print describe tables to console
    print("\n" + "=" * 60)
    print("PROFESSOR-FORMAT DESCRIPTIVE STATISTICS")
    print("=" * 60)
    for stock in ["AMZN", "GOOG", "INTC", "MSFT"]:
        print_describe_tables(per_min, stock)

    # 5. Generate plots
    print("\nGenerating plots ...")

    plot_alpha_journey(
        milestones, v9_train_df, v9_hold_df,
        os.path.join(out_dir, "p01_alpha_journey.png"))

    plot_buysell_decomposition(
        milestones, v9_train_df,
        os.path.join(out_dir, "p02_buysell_decomposition.png"))

    plot_perstock_progression(
        milestones, v9_train_df,
        os.path.join(out_dir, "p03_perstock_progression.png"))

    plot_v9_walkforward(
        pipe, table,
        os.path.join(out_dir, "p04_walkforward_validation.png"))

    plot_retention_flags(
        pipe,
        os.path.join(out_dir, "p05_retention_flags.png"))

    plot_improvement_distributions(
        per_min,
        os.path.join(out_dir, "p06_improvement_distributions.png"))

    plot_describe_tables(
        per_min,
        os.path.join(out_dir, "p07_describe_tables.png"))

    plot_twap_comparison(
        per_min,
        os.path.join(out_dir, "p08_twap_comparison.png"))

    plot_v9_alpha_per_stock(
        pipe,
        os.path.join(out_dir, "p09_v9_alpha_per_stock.png"))

    print(f"\nAll plots saved to {out_dir}/")
    print("Done.")
    return pipe, table, per_min


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    train_dir   = sys.argv[1] if len(sys.argv) > 1 else "./1. Project Files"
    project_dir = sys.argv[2] if len(sys.argv) > 2 else "./1. Project Files"
    out_dir     = sys.argv[3] if len(sys.argv) > 3 else "./1. Project Files/v9_presentation"

    run_presentation(train_dir, project_dir, out_dir, verbose=False)
