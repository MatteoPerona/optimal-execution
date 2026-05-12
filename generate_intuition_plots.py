"""Generate two intuition plots for the strategy report."""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.config import DEFAULT_CONFIG
from utils.preprocessing import load_all_stocks
from utils.signals import compute_mid_change_at_horizon

config = DEFAULT_CONFIG.copy()
data, frames, archetypes = load_all_stocks(config)

# ── Plot 1: OI predicts future price direction ──────────────────────────────

# Compute 10s forward mid-price change per stock
for ticker in config['stocks']:
    mask = data['ticker'] == ticker
    sub = data.loc[mask].reset_index(drop=True)
    data.loc[mask, 'dm_10s'] = compute_mid_change_at_horizon(sub, 10)
    med_spr = data.loc[mask, 'spread'].median()
    data.loc[mask, 'dm_10s_norm'] = data.loc[mask, 'dm_10s'] / med_spr

# Use GOOG minute 35820: OI rises from 0.08 to 0.90, mid follows by +1.6 spreads
df_goog = frames['GOOG']
grp = df_goog[df_goog['minute_start'] == 35820].copy()
t = grp['t_elapsed'].values
oi = grp['oi'].values
mid = grp['mid'].values
med_spr = grp['spread'].median()

fig, ax1 = plt.subplots(figsize=(10, 5))

# Mid-price on left axis
color_mid = '#2c3e50'
ax1.plot(t, mid, color=color_mid, lw=1.2, label='Mid-Price')
ax1.set_xlabel('Time Within Minute (seconds)', fontsize=12)
ax1.set_ylabel('Mid-Price ($)', fontsize=12, color=color_mid)
ax1.tick_params(axis='y', labelcolor=color_mid)

# OI on right axis
ax2 = ax1.twinx()
color_oi = '#e67e22'
ax2.fill_between(t, 0.5, oi, where=(oi > 0.5), alpha=0.3, color='#2ecc71', label='Buy pressure (OI > 0.5)')
ax2.fill_between(t, 0.5, oi, where=(oi < 0.5), alpha=0.3, color='#e74c3c', label='Sell pressure (OI < 0.5)')
ax2.plot(t, oi, color=color_oi, lw=0.8, alpha=0.7)
ax2.set_ylabel('Order-Book Imbalance', fontsize=12, color=color_oi)
ax2.tick_params(axis='y', labelcolor=color_oi)
ax2.axhline(0.5, color='gray', lw=0.5, ls='--')
ax2.set_ylim(0, 1)

# Annotations in clear areas
ax1.text(0.03, 0.05, f'OI starts low (0.08)\nPrice = ${mid[0]:.2f}',
         transform=ax1.transAxes, fontsize=9, va='bottom',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))

ax1.text(0.97, 0.95, f'OI rises to 0.90\nPrice = ${mid[-1]:.2f}  (+${mid[-1]-mid[0]:.2f})',
         transform=ax1.transAxes, fontsize=9, va='top', ha='right',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=9,
           framealpha=0.9)

fig.suptitle('GOOG: Order-Book Imbalance Leads Price Movement', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/intuition_oi_predicts_price.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: assets/intuition_oi_predicts_price.png")


# ── Plot 2: Cheaper to cross a tight spread ─────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, (ticker, arch) in zip(axes, [('MSFT', 'penny'), ('GOOG', 'wide')]):
    sub = frames[ticker]
    # Take a 2-minute window for visual clarity
    start = sub['seconds'].iloc[0]
    window = sub[(sub['seconds'] >= start + 120) & (sub['seconds'] < start + 240)].copy()

    t = window['seconds'] - window['seconds'].iloc[0]
    ask = window['AskPrice_1'].values
    bid = window['BidPrice_1'].values
    spread = ask - bid
    med_spr = np.median(spread)

    # Plot bid/ask
    ax.fill_between(t, bid, ask, alpha=0.25, color='steelblue', label='Spread (cost to cross)')
    ax.plot(t, ask, color='#e74c3c', lw=0.8, label='Ask (buy price)')
    ax.plot(t, bid, color='#2ecc71', lw=0.8, label='Bid (sell price)')

    # Find tight and wide moments
    tight_idx = np.argmin(spread)
    wide_idx = np.argmax(spread)

    price_range = ask.max() - bid.min()
    y_top = ask.max() + price_range * 0.15

    # Place both labels above the price action with arrows pointing down
    ax.annotate(f'Tight: ${spread[tight_idx]:.3f}',
                xy=(t.iloc[tight_idx], (ask[tight_idx] + bid[tight_idx]) / 2),
                xytext=(t.iloc[tight_idx], y_top),
                fontsize=9, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#2ecc71', alpha=0.9),
                color='#2ecc71')

    # Offset wide label horizontally if too close to tight label
    wide_x = t.iloc[wide_idx]
    tight_x = t.iloc[tight_idx]
    if abs(wide_x - tight_x) < 20:
        text_x = wide_x + 20
    else:
        text_x = wide_x

    ax.annotate(f'Wide: ${spread[wide_idx]:.3f}',
                xy=(t.iloc[wide_idx], (ask[wide_idx] + bid[wide_idx]) / 2),
                xytext=(text_x, y_top + price_range * 0.08),
                fontsize=9, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#e74c3c', alpha=0.9),
                color='#e74c3c')

    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Price ($)', fontsize=11)
    ax.set_title(f'{ticker} ({arch.title()}-Spread)  —  median spread = ${med_spr:.3f}', fontsize=12)
    ax.legend(fontsize=9, loc='lower left')

    # Extend y-axis to make room for labels
    ax.set_ylim(bid.min() - price_range * 0.05, y_top + price_range * 0.2)

fig.suptitle('Executing When the Spread Is Tight Reduces Crossing Cost', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/intuition_spread_cost.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: assets/intuition_spread_cost.png")
