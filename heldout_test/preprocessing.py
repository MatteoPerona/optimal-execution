import numpy as np
import pandas as pd

from utils.preprocessing import detect_archetype


def _filename_for_split(ticker, split):
    """Return the canonical CSV name for a train/test split."""
    if split not in {'train', 'test'}:
        raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'test'.")
    return f'{ticker}_5levels_{split}.csv'


def load_and_preprocess(ticker, data_dir, split):
    """Load one stock CSV for the requested split and compute core features."""
    filename = _filename_for_split(ticker, split)
    df = pd.read_csv(f'{data_dir}/{filename}')

    parts = df['Time'].str.split(':', expand=True)
    df['seconds'] = (
        parts[0].astype(float) * 3600
        + parts[1].astype(float) * 60
        + parts[2].astype(float)
    )

    df['minute_start'] = np.floor(df['seconds'] / 60) * 60
    df['mid'] = (df['BidPrice_1'] + df['AskPrice_1']) / 2
    df['spread'] = df['AskPrice_1'] - df['BidPrice_1']
    df['oi'] = df['BidSize_1'] / (df['BidSize_1'] + df['AskSize_1'])
    df['t_elapsed'] = df['seconds'] - df['minute_start']
    df['ticker'] = ticker
    return df


def load_all_stocks(config, split, archetypes=None):
    """Load all stocks for the requested split.

    Parameters
    ----------
    config : dict
        Uses ``stocks``, ``train_data_dir``, ``test_data_dir``,
        and ``penny_spread_cutoff``.
    split : {'train', 'test'}
        Which directory/filename convention to load.
    archetypes : dict or None
        If provided, map these train-time archetypes onto the loaded rows.
        If omitted, infer archetypes from the loaded data.
    """
    stocks = config['stocks']
    if split == 'train':
        data_dir = config['train_data_dir']
    elif split == 'test':
        data_dir = config['test_data_dir']
    else:
        raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'test'.")

    cutoff = config['penny_spread_cutoff']
    frames = {ticker: load_and_preprocess(ticker, data_dir, split) for ticker in stocks}
    data = pd.concat(frames.values(), ignore_index=True)

    if archetypes is None:
        archetypes = {ticker: detect_archetype(frames[ticker], cutoff) for ticker in stocks}
    else:
        missing = [ticker for ticker in stocks if ticker not in archetypes]
        if missing:
            raise KeyError(
                f"Missing archetypes for tickers: {', '.join(sorted(missing))}"
            )

    data['archetype'] = data['ticker'].map(archetypes)

    twap_ask = data.groupby(['ticker', 'minute_start'])['AskPrice_1'].mean().rename('twap_ask')
    twap_bid = data.groupby(['ticker', 'minute_start'])['BidPrice_1'].mean().rename('twap_bid')
    data = data.merge(twap_ask, on=['ticker', 'minute_start'])
    data = data.merge(twap_bid, on=['ticker', 'minute_start'])

    return data, frames, archetypes
