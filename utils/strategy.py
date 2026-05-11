import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from .signals import get_signal_fn


# ---------------------------------------------------------------------------
# Base class — subclass this to create a new strategy
# ---------------------------------------------------------------------------

class BaseStrategy:
    """Interface for execution strategies.

    Subclasses must implement `decide()`.  Everything else (grid search,
    backtesting, comparison) works automatically for any strategy that
    follows this interface.

    Parameters
    ----------
    params : dict
        Strategy-specific parameters (e.g. {'theta_imb': 0.9, 'theta_spread': 0.01}).
    signal_fn : str or callable
        Signal function name (from SIGNAL_REGISTRY) or a callable.
        Must accept a minute DataFrame and return a 1D array in [0, 1]
        where higher = stronger buy signal.
    """

    name = 'base'  # override in subclasses for display

    def __init__(self, params, signal_fn='oi'):
        self.params = params
        self.signal_fn = get_signal_fn(signal_fn)

    def decide(self, signal, spread, side):
        """Return the tick index to execute at, or None for last-tick fallback.

        Parameters
        ----------
        signal : np.ndarray  — signal values for each tick (higher = buy pressure)
        spread : np.ndarray  — spread at each tick
        side : str           — 'buy' or 'sell'

        Returns
        -------
        int or None — tick index, or None to execute at last tick.
        """
        raise NotImplementedError

    def execute_minute(self, grp, side):
        """Run the strategy on one minute of data. Returns exec_price.

        You normally don't need to override this — just implement decide().
        """
        signal_raw = self.signal_fn(grp)
        spread = grp['spread'].values
        ask = grp['AskPrice_1'].values
        bid = grp['BidPrice_1'].values

        # Flip signal for sells: high raw signal = buy pressure,
        # so for sells we want low raw signal = sell pressure
        signal = signal_raw if side == 'buy' else 1.0 - signal_raw

        idx = self.decide(signal, spread, side)
        if idx is None:
            idx = len(ask) - 1
        return ask[idx] if side == 'buy' else bid[idx]

    # -- Param grid (override to define your search space) --

    @classmethod
    def param_grid(cls, archetype):
        """Return a dict of {param_name: np.ndarray} defining the search grid.

        Override in subclasses.  The grid search will try every combination.
        """
        raise NotImplementedError

    @classmethod
    def from_grid_point(cls, signal_fn='oi', **kwargs):
        """Construct an instance from individual grid-search param values."""
        return cls(params=kwargs, signal_fn=signal_fn)


# ---------------------------------------------------------------------------
# Built-in strategy: fixed OI + spread threshold
# ---------------------------------------------------------------------------

class OIThresholdStrategy(BaseStrategy):
    """Fire when signal > theta_imb AND spread <= theta_spread."""

    name = 'OI Threshold'

    def decide(self, signal, spread, side, **kwargs):
        th_imb = self.params['theta_imb']
        th_spread = self.params['theta_spread']
        for i in range(len(signal)):
            if signal[i] > th_imb and spread[i] <= th_spread:
                return i
        return None

    @classmethod
    def param_grid(cls, archetype):
        if archetype == 'penny':
            return {
                'theta_imb': np.linspace(0.52, 0.95, 30),
                'theta_spread': np.linspace(0.005, 0.03, 20),
            }
        else:
            return {
                'theta_imb': np.linspace(0.52, 0.95, 30),
                'theta_spread': np.linspace(0.05, 0.50, 20),
            }


# ---------------------------------------------------------------------------
# Grid search — works with any BaseStrategy subclass
# ---------------------------------------------------------------------------

def precompute_minute_data(subset, signal_fn):
    """Pre-extract arrays per minute for fast grid search."""
    sig_fn = get_signal_fn(signal_fn)
    minutes = []
    for (ticker, minute), grp in subset.groupby(['ticker', 'minute_start']):
        if len(grp) < 5:
            continue
        minutes.append({
            'signal': sig_fn(grp),
            'spread': grp['spread'].values,
            'ask': grp['AskPrice_1'].values,
            'bid': grp['BidPrice_1'].values,
            'twap_ask': grp['twap_ask'].iloc[0],
            'twap_bid': grp['twap_bid'].iloc[0],
        })
    return minutes


def _score_strategy_on_precomputed(strategy_cls, params, precomputed, side, signal_fn):
    """Score a single parameter set on precomputed data. Returns mean improvement."""
    strat = strategy_cls.from_grid_point(signal_fn=signal_fn, **params)
    total = 0.0
    n = len(precomputed)
    if n == 0:
        return 0.0

    has_time = 't_elapsed' in precomputed[0]

    for m in precomputed:
        signal = m['signal'] if side == 'buy' else 1.0 - m['signal']
        if has_time:
            idx = strat.decide(signal, m['spread'], side, t_elapsed=m['t_elapsed'])
        else:
            idx = strat.decide(signal, m['spread'], side)
        if idx is None:
            idx = len(m['ask']) - 1

        price = m['ask'][idx] if side == 'buy' else m['bid'][idx]
        twap = m['twap_ask'] if side == 'buy' else m['twap_bid']
        if side == 'buy':
            total += twap - price
        else:
            total += price - twap

    return total / n


def grid_search(strategy_cls, precomputed, archetype, side, signal_fn='oi'):
    """Run grid search for a strategy class on one archetype and side.

    Returns (param_names, grid_arrays, scores_nd) where scores_nd is an
    N-dimensional array indexed by the grid axes.
    """
    grid = strategy_cls.param_grid(archetype)
    param_names = list(grid.keys())
    grid_arrays = [grid[k] for k in param_names]

    # Build flat list of all combos
    from itertools import product
    shape = tuple(len(a) for a in grid_arrays)
    scores = np.zeros(shape)

    for idx in np.ndindex(*shape):
        params = {name: arr[i] for name, arr, i in zip(param_names, grid_arrays, idx)}
        scores[idx] = _score_strategy_on_precomputed(
            strategy_cls, params, precomputed, side, signal_fn
        )

    return param_names, grid_arrays, scores


def smooth_and_select(scores, grid_arrays, param_names, smooth_size=3):
    """Smooth a grid-search surface and return best params.

    Returns (best_params_dict, best_score, smoothed_scores).
    """
    smoothed = scores.copy()
    for axis in range(smoothed.ndim):
        smoothed = uniform_filter1d(smoothed, smooth_size, axis=axis)

    best_idx = np.unravel_index(smoothed.argmax(), smoothed.shape)
    best_params = {name: arr[i] for name, arr, i in zip(param_names, grid_arrays, best_idx)}
    return best_params, smoothed[best_idx], smoothed


def fit_strategy(strategy_cls, train_data, config, signal_fn='oi'):
    """Fit a strategy's parameters via grid search + smoothing.

    Returns
    -------
    fitted : dict — {archetype: best_params_dict}
    details : dict — {(archetype, side): (best_params, best_score, smoothed_surface, param_names, grid_arrays)}
    """
    smooth_size = config.get('smooth_size', 3)

    fitted = {}
    details = {}

    for arch in ['penny', 'wide']:
        arch_data = train_data[train_data['archetype'] == arch]
        # Use time-aware precompute if strategy needs t_elapsed
        needs_time = hasattr(strategy_cls, 'execute_minute') and strategy_cls is not OIThresholdStrategy
        if needs_time:
            pre = _precompute_with_time(arch_data, signal_fn)
        else:
            pre = precompute_minute_data(arch_data, signal_fn)

        side_params = {}
        for side in ['buy', 'sell']:
            param_names, grid_arrays, raw_scores = grid_search(
                strategy_cls, pre, arch, side, signal_fn
            )
            best_params, best_score, smoothed = smooth_and_select(
                raw_scores, grid_arrays, param_names, smooth_size
            )
            side_params[side] = best_params
            details[(arch, side)] = (best_params, best_score, smoothed, param_names, grid_arrays)

        # Average buy/sell params
        all_keys = list(side_params['buy'].keys())
        fitted[arch] = {
            k: (side_params['buy'][k] + side_params['sell'][k]) / 2 for k in all_keys
        }

    return fitted, details


# ---------------------------------------------------------------------------
# Slope strategy: threshold = theta_imb + slope * (t / 60)
# ---------------------------------------------------------------------------

class SlopeStrategy(BaseStrategy):
    """Threshold changes linearly over the minute.

    theta(t) = theta_imb + slope * (t_elapsed / 60)
    Clamped to [0, 1].

    slope > 0 → threshold rises (pickier late)
    slope < 0 → threshold decays (more lenient late)
    slope = 0 → equivalent to fixed OIThresholdStrategy

    params: theta_imb, slope, theta_spread
    """

    name = 'Slope'

    def decide(self, signal, spread, side, t_elapsed=None):
        th_base = self.params['theta_imb']
        slope = self.params['slope']
        th_spread = self.params['theta_spread']

        for i in range(len(signal)):
            th = np.clip(th_base + slope * (t_elapsed[i] / 60.0), 0.0, 1.0)
            if signal[i] > th and spread[i] <= th_spread:
                return i
        return None

    def execute_minute(self, grp, side):
        signal_raw = self.signal_fn(grp)
        spread = grp['spread'].values
        ask = grp['AskPrice_1'].values
        bid = grp['BidPrice_1'].values
        t_elapsed = grp['t_elapsed'].values

        signal = signal_raw if side == 'buy' else 1.0 - signal_raw
        idx = self.decide(signal, spread, side, t_elapsed=t_elapsed)
        if idx is None:
            idx = len(ask) - 1
        return ask[idx] if side == 'buy' else bid[idx]

    @classmethod
    def param_grid(cls, archetype):
        return {
            'theta_imb': np.linspace(0.52, 0.95, 25),
            'slope': np.linspace(-0.4, 0.4, 25),
        }


def _precompute_with_time(subset, signal_fn):
    """Like precompute_minute_data but includes t_elapsed."""
    sig_fn = get_signal_fn(signal_fn)
    minutes = []
    for (ticker, minute), grp in subset.groupby(['ticker', 'minute_start']):
        if len(grp) < 5:
            continue
        minutes.append({
            'signal': sig_fn(grp),
            'spread': grp['spread'].values,
            'ask': grp['AskPrice_1'].values,
            'bid': grp['BidPrice_1'].values,
            't_elapsed': grp['t_elapsed'].values,
            'twap_ask': grp['twap_ask'].iloc[0],
            'twap_bid': grp['twap_bid'].iloc[0],
        })
    return minutes


# Strategies that need t_elapsed in their precomputed data
_TIME_STRATEGIES = set()


def _needs_time(strategy_cls):
    return strategy_cls in _TIME_STRATEGIES or hasattr(strategy_cls, '_needs_time')


# ---------------------------------------------------------------------------
# Two-window strategy with greedy sequential fitting
# ---------------------------------------------------------------------------

WINDOW_SPLIT = 30.0  # seconds — boundary between window 1 and window 2

class TwoWindowStrategy(BaseStrategy):
    """Different imbalance thresholds for first and second half of the minute.

    params: theta_imb_w1, theta_imb_w2, theta_spread
    """

    name = 'Two-Window'

    def decide(self, signal, spread, side, t_elapsed=None):
        th_w1 = self.params['theta_imb_w1']
        th_w2 = self.params['theta_imb_w2']
        th_spread = self.params['theta_spread']

        for i in range(len(signal)):
            th = th_w1 if t_elapsed[i] < WINDOW_SPLIT else th_w2
            if signal[i] > th and spread[i] <= th_spread:
                return i
        return None

    def execute_minute(self, grp, side):
        signal_raw = self.signal_fn(grp)
        spread = grp['spread'].values
        ask = grp['AskPrice_1'].values
        bid = grp['BidPrice_1'].values
        t_elapsed = grp['t_elapsed'].values

        signal = signal_raw if side == 'buy' else 1.0 - signal_raw
        idx = self.decide(signal, spread, side, t_elapsed=t_elapsed)
        if idx is None:
            idx = len(ask) - 1
        return ask[idx] if side == 'buy' else bid[idx]


def _precompute_with_time(subset, signal_fn):
    """Like precompute_minute_data but includes t_elapsed."""
    sig_fn = get_signal_fn(signal_fn)
    minutes = []
    for (ticker, minute), grp in subset.groupby(['ticker', 'minute_start']):
        if len(grp) < 5:
            continue
        minutes.append({
            'signal': sig_fn(grp),
            'spread': grp['spread'].values,
            'ask': grp['AskPrice_1'].values,
            'bid': grp['BidPrice_1'].values,
            't_elapsed': grp['t_elapsed'].values,
            'twap_ask': grp['twap_ask'].iloc[0],
            'twap_bid': grp['twap_bid'].iloc[0],
        })
    return minutes


def _score_two_window(precomputed, th_imb_w1, th_imb_w2, th_spread, side):
    """Score a (th_imb_w1, th_imb_w2, th_spread) combo."""
    total = 0.0
    n = len(precomputed)
    if n == 0:
        return 0.0
    for m in precomputed:
        signal = m['signal'] if side == 'buy' else 1.0 - m['signal']
        t_el = m['t_elapsed']
        idx = None
        for i in range(len(signal)):
            th = th_imb_w1 if t_el[i] < WINDOW_SPLIT else th_imb_w2
            if signal[i] > th and m['spread'][i] <= th_spread:
                idx = i
                break
        if idx is None:
            idx = len(signal) - 1
        price = m['ask'][idx] if side == 'buy' else m['bid'][idx]
        twap = m['twap_ask'] if side == 'buy' else m['twap_bid']
        total += (twap - price) if side == 'buy' else (price - twap)
    return total / n


def fit_two_window(train_data, config, signal_fn='oi', base_fitted=None):
    """Greedy sequential fit for TwoWindowStrategy.

    Step 1: Fix theta_spread from the base OIThreshold fit.
    Step 2: 1D grid search for theta_imb_w1 (scored on all minutes).
    Step 3: 1D grid search for theta_imb_w2 (scored with w1 fixed).

    Returns fitted dict and details.
    """
    smooth_size = config.get('smooth_size', 3)
    imb_grid = np.linspace(0.52, 0.95, 30)

    if base_fitted is None:
        base_fitted, _ = fit_strategy(OIThresholdStrategy, train_data, config, signal_fn)

    fitted = {}
    details = {}

    for arch in ['penny', 'wide']:
        th_spread = base_fitted[arch]['theta_spread']
        pre = _precompute_with_time(
            train_data[train_data['archetype'] == arch], signal_fn
        )

        side_params = {}
        for side in ['buy', 'sell']:
            # Step 1: grid search theta_imb_w1 with theta_imb_w2 set very high
            # (effectively: only window 1 can fire, fallback to last tick otherwise)
            w1_scores = np.array([
                _score_two_window(pre, th1, 999.0, th_spread, side)
                for th1 in imb_grid
            ])
            w1_smoothed = uniform_filter1d(w1_scores, smooth_size)
            best_w1 = imb_grid[w1_smoothed.argmax()]

            # Step 2: fix w1, grid search theta_imb_w2
            w2_scores = np.array([
                _score_two_window(pre, best_w1, th2, th_spread, side)
                for th2 in imb_grid
            ])
            w2_smoothed = uniform_filter1d(w2_scores, smooth_size)
            best_w2 = imb_grid[w2_smoothed.argmax()]

            best_score = _score_two_window(pre, best_w1, best_w2, th_spread, side)
            side_params[side] = {
                'theta_imb_w1': best_w1,
                'theta_imb_w2': best_w2,
                'theta_spread': th_spread,
            }
            details[(arch, side)] = {
                'params': side_params[side],
                'score': best_score,
                'w1_scores': w1_smoothed,
                'w2_scores': w2_smoothed,
                'imb_grid': imb_grid,
            }

        # Average buy/sell
        fitted[arch] = {
            'theta_imb_w1': (side_params['buy']['theta_imb_w1'] + side_params['sell']['theta_imb_w1']) / 2,
            'theta_imb_w2': (side_params['buy']['theta_imb_w2'] + side_params['sell']['theta_imb_w2']) / 2,
            'theta_spread': th_spread,
        }

    return fitted, details


def _score_slope(precomputed, th_imb, slope_val, th_spread, side):
    """Score a (theta_imb, slope, theta_spread) combo."""
    total = 0.0
    n = len(precomputed)
    if n == 0:
        return 0.0
    for m in precomputed:
        signal = m['signal'] if side == 'buy' else 1.0 - m['signal']
        t_el = m['t_elapsed']
        idx = None
        for i in range(len(signal)):
            th = np.clip(th_imb + slope_val * (t_el[i] / 60.0), 0.0, 1.0)
            if signal[i] > th and m['spread'][i] <= th_spread:
                idx = i
                break
        if idx is None:
            idx = len(signal) - 1
        price = m['ask'][idx] if side == 'buy' else m['bid'][idx]
        twap = m['twap_ask'] if side == 'buy' else m['twap_bid']
        total += (twap - price) if side == 'buy' else (price - twap)
    return total / n


def fit_slope(train_data, config, signal_fn='oi', base_fitted=None):
    """Fit SlopeStrategy: fix theta_spread from base fit, 2D search over (theta_imb, slope).

    Returns fitted dict and details.
    """
    smooth_size = config.get('smooth_size', 3)
    imb_grid = np.linspace(0.52, 0.95, 25)
    slope_grid = np.linspace(-0.4, 0.4, 25)

    if base_fitted is None:
        base_fitted, _ = fit_strategy(OIThresholdStrategy, train_data, config, signal_fn)

    fitted = {}
    details = {}

    for arch in ['penny', 'wide']:
        th_spread = base_fitted[arch]['theta_spread']
        pre = _precompute_with_time(
            train_data[train_data['archetype'] == arch], signal_fn
        )

        side_params = {}
        for side in ['buy', 'sell']:
            scores = np.zeros((len(imb_grid), len(slope_grid)))
            for i, th_imb in enumerate(imb_grid):
                for j, sl in enumerate(slope_grid):
                    scores[i, j] = _score_slope(pre, th_imb, sl, th_spread, side)

            # Smooth and select
            smoothed = uniform_filter1d(
                uniform_filter1d(scores, smooth_size, axis=0), smooth_size, axis=1
            )
            best_idx = np.unravel_index(smoothed.argmax(), smoothed.shape)
            best_imb = imb_grid[best_idx[0]]
            best_slope = slope_grid[best_idx[1]]

            side_params[side] = {
                'theta_imb': best_imb,
                'slope': best_slope,
                'theta_spread': th_spread,
            }
            details[(arch, side)] = {
                'params': side_params[side],
                'score': smoothed[best_idx],
                'smoothed': smoothed,
                'imb_grid': imb_grid,
                'slope_grid': slope_grid,
            }

        fitted[arch] = {
            'theta_imb': (side_params['buy']['theta_imb'] + side_params['sell']['theta_imb']) / 2,
            'slope': (side_params['buy']['slope'] + side_params['sell']['slope']) / 2,
            'theta_spread': th_spread,
        }

    return fitted, details


# ---------------------------------------------------------------------------
# run_experiment: strategy + config → results
# ---------------------------------------------------------------------------

def run_experiment(strategy_cls, config, signal_fn='oi', verbose=True,
                   _shared=None):
    """End-to-end: load data, fit strategy, backtest, return results.

    Parameters
    ----------
    strategy_cls : subclass of BaseStrategy
    config : dict (see utils/config.py for defaults)
    signal_fn : str or callable — which signal to use
    verbose : bool — print progress
    _shared : dict or None
        Pass a dict to cache and reuse expensive computations (data loading,
        base fit) across multiple run_experiment calls.  The dict is mutated
        in place — pass the same object to each call.

    Returns
    -------
    dict with keys:
        'fitted'     — {archetype: params_dict}
        'details'    — grid search details
        'test_all'   — DataFrame of per-minute backtest results
        'data'       — full tick DataFrame
        'frames'     — {ticker: DataFrame}
        'archetypes' — {ticker: archetype}
    """
    from .preprocessing import load_all_stocks, train_test_split
    from .evaluation import evaluate_both_sides

    if _shared is None:
        _shared = {}

    if verbose:
        print(f"Running experiment: {strategy_cls.name} (signal={signal_fn})")

    # Cache data loading
    if 'data' not in _shared:
        data, frames, archetypes = load_all_stocks(config)
        train_data, test_data = train_test_split(data, config)
        _shared['data'] = data
        _shared['frames'] = frames
        _shared['archetypes'] = archetypes
        _shared['train_data'] = train_data
        _shared['test_data'] = test_data
    else:
        data = _shared['data']
        frames = _shared['frames']
        archetypes = _shared['archetypes']
        train_data = _shared['train_data']
        test_data = _shared['test_data']

    if verbose:
        for t in config['stocks']:
            print(f"  {t}: {archetypes[t]}")
        print(f"  Train: {train_data['minute_start'].nunique()} min, "
              f"Test: {test_data['minute_start'].nunique()} min")

    if verbose:
        print("  Fitting parameters...")

    # Cache base fit for strategies that depend on it
    base_fitted = _shared.get('base_fitted')

    if strategy_cls is TwoWindowStrategy:
        if base_fitted is None:
            base_fitted, _ = fit_strategy(OIThresholdStrategy, train_data, config, signal_fn)
            _shared['base_fitted'] = base_fitted
        fitted, details = fit_two_window(train_data, config, signal_fn, base_fitted=base_fitted)
    elif strategy_cls is SlopeStrategy:
        if base_fitted is None:
            base_fitted, _ = fit_strategy(OIThresholdStrategy, train_data, config, signal_fn)
            _shared['base_fitted'] = base_fitted
        fitted, details = fit_slope(train_data, config, signal_fn, base_fitted=base_fitted)
    else:
        fitted, details = fit_strategy(strategy_cls, train_data, config, signal_fn)
        # Cache if this is the base strategy
        if strategy_cls is OIThresholdStrategy:
            _shared['base_fitted'] = fitted

    if verbose:
        print("  Done.")
    if verbose:
        for arch, params in fitted.items():
            param_str = ', '.join(f'{k}={v:.4f}' for k, v in params.items())
            print(f"    {arch}: {param_str}")

    if verbose:
        print("  Backtesting...")
    test_all = evaluate_both_sides(
        strategy_cls, test_data, fitted, signal_fn
    )

    return {
        'fitted': fitted,
        'details': details,
        'test_all': test_all,
        'data': data,
        'frames': frames,
        'archetypes': archetypes,
        'train_data': train_data,
        'test_data': test_data,
    }
