import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

NAVY   = "#002060"
BLUE   = "#1565C0"
RED    = "#C62828"
GREEN  = "#2E7D32"
PURPLE = "#6A1B9A"
PALETTE = [NAVY, BLUE, RED, GREEN, PURPLE, "#E65100", "#00695C"]


def plot_results_bar(table, out):
    stocks = table["Stock"].tolist()
    x      = np.arange(len(stocks))
    w      = 0.25

    train   = table["Train Improv (%)"].values.astype(float)
    holdout = table["Holdout Improv (%)"].values.astype(float)
    has_test = table["Test Improv (%)"].notna().any()

    fig, ax = plt.subplots(figsize=(10, 6))
    if has_test:
        ax.bar(x - w,       train,   w, label="Train",   color=NAVY)
        ax.bar(x,           holdout, w, label="Holdout",  color=BLUE)
        test = table["Test Improv (%)"].values.astype(float)
        ax.bar(x + w,       test,    w, label="Test",     color=GREEN)
    else:
        ax.bar(x - w / 2,   train,   w, label="Train",   color=NAVY)
        ax.bar(x + w / 2,   holdout, w, label="Holdout",  color=BLUE)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stocks, fontsize=11)
    ax.set_ylabel("Improvement vs TWAP (%)", fontsize=11)
    ax.set_title("Strategy Performance: Train vs Holdout vs Test", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_strategy_distribution(pipeline, out):
    stocks     = list(pipeline.keys())
    strategies = [pipeline[s]["best"]["strategy_name"] for s in stocks]
    unique_s   = sorted(set(strategies))
    cmap       = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(unique_s)}
    colors     = [cmap[s] for s in strategies]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(stocks, [1] * len(stocks), color=colors, edgecolor="white")

    for bar, strat in zip(bars, strategies):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.5, strat,
                ha="center", va="center", color="white", fontsize=9,
                fontweight="bold", rotation=30)

    ax.set_ylim(0, 1.6)
    ax.set_yticks([])
    ax.set_title("Strategy Selected per Stock", fontsize=13, fontweight="bold")
    legend_els = [Patch(facecolor=cmap[s], label=s) for s in unique_s]
    ax.legend(handles=legend_els, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_buy_sell_improvement(table, out):
    stocks = table["Stock"].tolist()
    x      = np.arange(len(stocks))
    w      = 0.35

    buy_a  = table["Train Buy α (bps)"].values.astype(float)
    sell_a = table["Train Sell α (bps)"].values.astype(float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w / 2, buy_a,  w, label="Buy α (bps)",  color=GREEN)
    ax.bar(x + w / 2, sell_a, w, label="Sell α (bps)", color=RED)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stocks, fontsize=11)
    ax.set_ylabel("Basis Points vs TWAP", fontsize=11)
    ax.set_title("Buy and Sell Alpha (bps) vs TWAP — Full Training Data", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_aapl_routing(pipeline, out):
    if "AAPL" not in pipeline:
        return

    selection = pipeline["AAPL"].get("selection", {})
    distances = selection.get("distances", {})
    nearest   = selection.get("nearest", "")

    if not distances:
        return

    stocks = list(distances.keys())
    dists  = [distances[s] for s in stocks]
    colors = [RED if s == nearest else BLUE for s in stocks]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(stocks, dists, color=colors, edgecolor="white")
    for bar, d in zip(bars, dists):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{d:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Fingerprint Distance (log-Euclidean)", fontsize=11)
    ax.set_title("AAPL Microstructure Fingerprint Distances\n(red = nearest neighbor)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_execution_curves(pipeline, stock, out):
    if stock not in pipeline:
        return

    data      = pipeline[stock]
    tw_f      = data.get("twap_full", {})
    al_f      = data.get("algo_full", {})
    buy_twap  = tw_f.get("buy_prices")
    buy_algo  = al_f.get("buy_prices")
    sell_twap = tw_f.get("sell_prices")
    sell_algo = al_f.get("sell_prices")

    if buy_twap is None:
        return

    n = min(200, len(buy_twap))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(n), buy_twap.values[:n],  color=NAVY,  label="TWAP", alpha=0.75, linewidth=1)
    ax1.plot(range(n), buy_algo.values[:n],  color=RED,   label="Algo", alpha=0.75, linewidth=1)
    ax1.set_title(f"{stock} — BUY Execution", fontsize=12)
    ax1.set_xlabel("Minute")
    ax1.set_ylabel("Execution Price")
    ax1.legend(fontsize=9)

    ax2.plot(range(n), sell_twap.values[:n], color=NAVY,  label="TWAP", alpha=0.75, linewidth=1)
    ax2.plot(range(n), sell_algo.values[:n], color=GREEN, label="Algo", alpha=0.75, linewidth=1)
    ax2.set_title(f"{stock} — SELL Execution", fontsize=12)
    ax2.set_xlabel("Minute")
    ax2.set_ylabel("Execution Price")
    ax2.legend(fontsize=9)

    fig.suptitle(f"{stock}: TWAP vs Algorithm (first {n} minutes)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def make_all_plots(pipeline, table, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    plot_results_bar(table,
                     os.path.join(out_dir, "p1_results_bar.png"))
    print("  Saved p1_results_bar.png")

    plot_strategy_distribution(pipeline,
                               os.path.join(out_dir, "p2_strategy_distribution.png"))
    print("  Saved p2_strategy_distribution.png")

    plot_buy_sell_improvement(table,
                              os.path.join(out_dir, "p3_buy_sell_alpha.png"))
    print("  Saved p3_buy_sell_alpha.png")

    plot_aapl_routing(pipeline,
                      os.path.join(out_dir, "p4_aapl_routing.png"))
    print("  Saved p4_aapl_routing.png")

    for stock in pipeline:
        fname = f"p5_{stock}_execution_curves.png"
        plot_execution_curves(pipeline, stock,
                              os.path.join(out_dir, fname))
        print(f"  Saved {fname}")
