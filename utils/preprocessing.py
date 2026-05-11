import numpy as np
import pandas as pd


def load_and_preprocess(ticker, data_dir='data'):
    """Load a single stock's CSV and compute core features.

    Adds columns: seconds, minute_start, mid, spread, oi, t_elapsed, ticker.
    """
    df = pd.read_csv(f'{data_dir}/{ticker}_5levels_train.csv')

    parts = df['Time'].str.split(':', expand=True)
    df['seconds'] = (parts[0].astype(float) * 3600 +
                     parts[1].astype(float) * 60 +
                     parts[2].astype(float))

    df['minute_start'] = np.floor(df['seconds'] / 60) * 60
    df['mid'] = (df['BidPrice_1'] + df['AskPrice_1']) / 2
    df['spread'] = df['AskPrice_1'] - df['BidPrice_1']
    df['oi'] = df['BidSize_1'] / (df['BidSize_1'] + df['AskSize_1'])
    df['t_elapsed'] = df['seconds'] - df['minute_start']
    df['ticker'] = ticker
    df['tod_bucket'] = pd.cut(
        df['seconds'],
        bins=[34200, 36000, 55800, 57600],
        labels=['open', 'mid', 'close'],
    )
    return df


def detect_archetype(df, cutoff=0.02):
    """Classify a stock as 'penny' or 'wide' spread from its first minute."""
    first_min = df['minute_start'].min()
    med_spread = df[df['minute_start'] == first_min]['spread'].median()
    return 'penny' if med_spread <= cutoff else 'wide'


def load_all_stocks(config):
    """Load all stocks, detect archetypes, add side-specific TWAP columns.

    Parameters
    ----------
    config : dict with keys 'stocks', 'data_dir', 'penny_spread_cutoff'.

    Returns
    -------
    data : DataFrame — concatenated tick data with twap_ask, twap_bid, archetype.
    frames : dict — {ticker: DataFrame}.
    archetypes : dict — {ticker: 'penny' | 'wide'}.
    """
    stocks = config['stocks']
    data_dir = config['data_dir']
    cutoff = config['penny_spread_cutoff']

    frames = {t: load_and_preprocess(t, data_dir) for t in stocks}
    data = pd.concat(frames.values(), ignore_index=True)

    archetypes = {t: detect_archetype(frames[t], cutoff) for t in stocks}
    data['archetype'] = data['ticker'].map(archetypes)

    twap_ask = data.groupby(['ticker', 'minute_start'])['AskPrice_1'].mean().rename('twap_ask')
    twap_bid = data.groupby(['ticker', 'minute_start'])['BidPrice_1'].mean().rename('twap_bid')
    data = data.merge(twap_ask, on=['ticker', 'minute_start'])
    data = data.merge(twap_bid, on=['ticker', 'minute_start'])

    return data, frames, archetypes


def train_test_split(data, config):
    """Split data by time into train and test sets."""
    all_minutes = sorted(data['minute_start'].unique())
    split_idx = int(len(all_minutes) * config['train_frac'])
    train_minutes = set(all_minutes[:split_idx])
    test_minutes = set(all_minutes[split_idx:])

    train_data = data[data['minute_start'].isin(train_minutes)].copy()
    test_data = data[data['minute_start'].isin(test_minutes)].copy()
    return train_data, test_data
