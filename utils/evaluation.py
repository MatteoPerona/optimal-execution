import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .signals import get_signal_fn


def evaluate_strategy(strategy_cls, test_data, fitted_params, signal_fn, side):
    """Backtest a fitted strategy and baselines on test data for one side.

    Returns a DataFrame with one row per (ticker, minute).
    """
    sig_fn = get_signal_fn(signal_fn)
    records = []
    columns = [
        'ticker', 'minute_start', 'archetype', 'twap',
        'med_spread', 'strategy', 'first_tick', 'last_tick',
    ]

    for (ticker, minute), grp in test_data.groupby(['ticker', 'minute_start']):
        if len(grp) < 5:
            continue
        arch = grp['archetype'].iloc[0]
        params = strategy_cls.lookup_params(fitted_params, grp)
        if params is None:
            continue
        twap_val = grp['twap_ask'].iloc[0] if side == 'buy' else grp['twap_bid'].iloc[0]
        ask = grp['AskPrice_1'].values
        bid = grp['BidPrice_1'].values
        med_spread = grp['spread'].median()

        strat = strategy_cls(params=params, signal_fn=sig_fn)
        strategy_price = strat.execute_minute(grp, side)
        first_price = ask[0] if side == 'buy' else bid[0]
        last_price = ask[-1] if side == 'buy' else bid[-1]

        sign = 1 if side == 'buy' else -1
        records.append({
            'ticker': ticker,
            'minute_start': minute,
            'archetype': arch,
            'twap': twap_val,
            'med_spread': med_spread,
            'strategy': sign * (twap_val - strategy_price),
            'first_tick': sign * (twap_val - first_price),
            'last_tick': sign * (twap_val - last_price),
        })

    return pd.DataFrame(records, columns=columns)


def evaluate_both_sides(strategy_cls, test_data, fitted_params, signal_fn='oi'):
    """Run evaluate_strategy for buy and sell, return concatenated results."""
    buy = evaluate_strategy(strategy_cls, test_data, fitted_params, signal_fn, 'buy')
    sell = evaluate_strategy(strategy_cls, test_data, fitted_params, signal_fn, 'sell')
    return pd.concat([buy, sell], ignore_index=True)


def print_results(test_all, strategy_name='Strategy'):
    """Print summary tables of strategy performance."""
    strategies = ['strategy', 'first_tick', 'last_tick']
    labels = [strategy_name, 'First Tick', 'Last Tick']

    print("=" * 70)
    print("Mean Improvement over TWAP ($)")
    print("=" * 70)

    for arch in ['penny', 'wide', 'all']:
        sub = test_all if arch == 'all' else test_all[test_all['archetype'] == arch]
        vals = '  '.join(f"{l}: ${sub[s].mean():+.6f}" for s, l in zip(strategies, labels))
        wr = (sub['strategy'] > 0).mean()
        print(f"  {arch.title():6s} (n={len(sub):4d}, win_rate={wr:.1%})  {vals}")

    print()
    print("=" * 70)
    print("Mean Improvement (spread units)")
    print("=" * 70)
    for arch in ['penny', 'wide']:
        sub = test_all[test_all['archetype'] == arch]
        med_spr = sub['med_spread'].median()
        print(f"\n  {arch.title()} (median spread = ${med_spr:.4f}):")
        for s, label in zip(strategies, labels):
            val = sub[s].mean() / med_spr
            print(f"    {label:20s}: {val:+.4f} spread units")


def plot_results(test_all, strategy_name='Strategy'):
    """Bar chart of strategy vs baselines, split by archetype."""
    strategies = ['strategy', 'first_tick', 'last_tick']
    labels = [strategy_name, 'First Tick', 'Last Tick']
    colors = ['#2ecc71', '#e74c3c', '#3498db']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, arch in zip(axes, ['penny', 'wide']):
        sub = test_all[test_all['archetype'] == arch]
        med_spr = sub['med_spread'].median()
        means = [sub[s].mean() / med_spr for s in strategies]
        bars = ax.bar(labels, means, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_ylabel('Mean Improvement (spread units)')
        ax.set_title(f'{arch.title()}-Spread Stocks')
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:+.3f}', ha='center',
                    va='bottom' if val >= 0 else 'top', fontsize=9)

    fig.suptitle(f'Test Set: {strategy_name} vs. Baselines', fontweight='bold')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# compare_strategies: side-by-side comparison of multiple strategies
# ---------------------------------------------------------------------------

def compare_strategies(experiments, archetypes=('penny', 'wide')):
    """Compare multiple strategy experiment results side by side.

    Parameters
    ----------
    experiments : list of (name, results_dict) tuples
        Each results_dict is the output of run_experiment().

    Returns
    -------
    comparison : DataFrame with one row per strategy and columns for each metric.
    """
    rows = []
    for name, res in experiments:
        test_all = res['test_all']
        for arch in list(archetypes) + ['all']:
            sub = test_all if arch == 'all' else test_all[test_all['archetype'] == arch]
            if len(sub) == 0:
                continue
            med_spr = sub['med_spread'].median()
            rows.append({
                'strategy': name,
                'archetype': arch,
                'mean_improvement_$': sub['strategy'].mean(),
                'mean_improvement_spreads': sub['strategy'].mean() / med_spr,
                'std_spreads': sub['strategy'].std() / med_spr,
                'win_rate': (sub['strategy'] > 0).mean(),
                'n_minutes': len(sub),
            })

    comparison = pd.DataFrame(rows)

    # Print
    print("=" * 90)
    print("STRATEGY COMPARISON")
    print("=" * 90)
    for arch in list(archetypes) + ['all']:
        sub = comparison[comparison['archetype'] == arch]
        if len(sub) == 0:
            continue
        print(f"\n  {arch.title()}:")
        print(f"    {'Strategy':25s} {'Mean (spr)':>12s} {'Std (spr)':>12s} {'Win Rate':>10s}")
        print(f"    {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
        for _, row in sub.iterrows():
            print(f"    {row['strategy']:25s} {row['mean_improvement_spreads']:+12.4f} "
                  f"{row['std_spreads']:12.4f} {row['win_rate']:10.1%}")

    # Plot
    fig, axes = plt.subplots(1, len(archetypes), figsize=(6 * len(archetypes), 5))
    if len(archetypes) == 1:
        axes = [axes]

    names = [name for name, _ in experiments]
    colors_list = plt.cm.Set2(np.linspace(0, 1, len(names)))

    for ax, arch in zip(axes, archetypes):
        sub = comparison[comparison['archetype'] == arch]
        vals = [sub.loc[sub['strategy'] == n, 'mean_improvement_spreads'].values[0]
                for n in names]
        bars = ax.bar(names, vals, color=colors_list, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_ylabel('Mean Improvement (spread units)')
        ax.set_title(f'{arch.title()}-Spread')
        ax.tick_params(axis='x', rotation=20)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:+.3f}', ha='center',
                    va='bottom' if val >= 0 else 'top', fontsize=9)

    fig.suptitle('Strategy Comparison', fontweight='bold')
    plt.tight_layout()
    plt.show()

    return comparison
