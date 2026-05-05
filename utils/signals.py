import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Signal functions: each takes a minute group and returns a 1D array [0, 1]
# where higher = stronger buy signal.  For sells, the framework flips it
# automatically (uses 1 - signal).
# ---------------------------------------------------------------------------

def oi_signal(grp):
    """Level-1 order-book imbalance: BidSize_1 / (BidSize_1 + AskSize_1)."""
    return grp['oi'].values


def weighted_oi_signal(grp):
    """Depth-weighted OI across all 5 book levels (closer levels weighted more)."""
    weights = np.array([5, 4, 3, 2, 1], dtype=float)
    bid_depth = sum(w * grp[f'BidSize_{i}'].values for i, w in enumerate(weights, 1))
    ask_depth = sum(w * grp[f'AskSize_{i}'].values for i, w in enumerate(weights, 1))
    total = bid_depth + ask_depth
    return np.where(total > 0, bid_depth / total, 0.5)


# Registry: add new signal functions here so they can be referenced by name
SIGNAL_REGISTRY = {
    'oi': oi_signal,
    'weighted_oi': weighted_oi_signal,
}


def get_signal_fn(name):
    """Look up a signal function by name."""
    if callable(name):
        return name
    if name not in SIGNAL_REGISTRY:
        available = ', '.join(SIGNAL_REGISTRY.keys())
        raise ValueError(f"Unknown signal '{name}'. Available: {available}")
    return SIGNAL_REGISTRY[name]


# ---------------------------------------------------------------------------
# Horizon analysis (diagnostic, not used by strategy at runtime)
# ---------------------------------------------------------------------------

def compute_mid_change_at_horizon(df, horizon):
    """Mid-price change `horizon` seconds forward. Call per-stock only."""
    seconds = df['seconds'].values
    mid = df['mid'].values
    future_idx = np.searchsorted(seconds, seconds + horizon, side='right') - 1
    future_idx = np.clip(future_idx, 0, len(df) - 1)
    return mid[future_idx] - mid


def add_horizon_columns(data, config):
    """Add mid-price change columns (raw and spread-normalized) per horizon."""
    stocks = config['stocks']
    horizons = config['horizons']

    for ticker in stocks:
        mask = data['ticker'] == ticker
        sub = data.loc[mask].reset_index(drop=True)
        for h in horizons:
            data.loc[mask, f'dm_{h}s'] = compute_mid_change_at_horizon(sub, h)

    for ticker in stocks:
        mask = data['ticker'] == ticker
        med_spread = data.loc[mask, 'spread'].median()
        for h in horizons:
            data.loc[mask, f'dm_{h}s_norm'] = data.loc[mask, f'dm_{h}s'] / med_spread


def plot_edge_vs_imbalance(data, config):
    """Mean mid-price change vs. OI bins, one curve per horizon, by archetype."""
    horizons = config['horizons']
    n_bins = config['n_bins']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, arch in zip(axes, ['penny', 'wide']):
        sub = data[data['archetype'] == arch].copy()
        sub['oi_bin'] = pd.qcut(sub['oi'], n_bins, labels=False, duplicates='drop')
        bin_centers = sub.groupby('oi_bin')['oi'].mean()

        for h in horizons:
            means = sub.groupby('oi_bin')[f'dm_{h}s_norm'].mean()
            ax.plot(bin_centers, means, marker='o', markersize=3, label=f'{h}s')

        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_xlabel('Order-Book Imbalance (BidSize / Total)')
        ax.set_title(f'{arch.title()}-Spread Stocks')
        ax.legend(title='Horizon')

    axes[0].set_ylabel('Mean Mid-Price Change (spread units)')
    fig.suptitle('Edge vs. Imbalance by Horizon', fontweight='bold')
    plt.tight_layout()
    plt.show()
