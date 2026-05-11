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

    @classmethod
    def lookup_params(cls, fitted_params, grp):
        """Select the fitted params to use for one minute group."""
        arch = grp['archetype'].iloc[0]
        tod_bucket = grp['tod_bucket'].iloc[0]
        return fitted_params.get((arch, tod_bucket), fitted_params.get((arch, 'mid')))

    @classmethod
    def fit_params(cls, train_data, config, signal_fn='oi'):
        """Fit params for this strategy class on the training data."""
        return _fit_bucketed_strategy(cls, train_data, config, signal_fn)


# ---------------------------------------------------------------------------
# Built-in strategy: fixed OI + spread threshold
# ---------------------------------------------------------------------------

class OIThresholdStrategy(BaseStrategy):
    """Fire when signal > theta_imb AND spread <= theta_spread."""

    name = 'OI Threshold'

    def decide(self, signal, spread, side):
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


class TimeVaryingOIThresholdStrategy(OIThresholdStrategy):
    """Scale crossing thresholds by the smoothed intraday spread profile."""

    name = 'Time-Varying OI Threshold'
    imb_sensitivity = 0.08
    spread_power = 1.0
    min_theta_imb = 0.50
    max_theta_imb = 0.99

    @classmethod
    def fit_params(cls, train_data, config, signal_fn='oi'):
        """Fit one baseline threshold pair, then scale it by minute of day."""
        smooth_size = config.get('smooth_size', 3)
        fitted = {}
        details = {}

        for arch in ['penny', 'wide']:
            arch_train = train_data[train_data['archetype'] == arch].copy()
            pre = precompute_minute_data(arch_train, signal_fn)
            if len(pre) < 5:
                continue

            base_params, side_details = _fit_params_for_precomputed(
                cls, pre, arch, signal_fn, smooth_size
            )

            minute_profile = (
                arch_train.assign(minute_of_day=(arch_train['minute_start'] // 60).astype(int))
                .groupby('minute_of_day')
                .agg(avg_spread=('spread', 'mean'))
                .reset_index()
                .sort_values('minute_of_day')
            )

            spread_curve = minute_profile['avg_spread'].to_numpy(dtype=float)
            if len(spread_curve) == 0:
                continue
            spread_curve = uniform_filter1d(spread_curve, smooth_size)
            spread_ratio = spread_curve / spread_curve.mean()

            minutes = minute_profile['minute_of_day'].to_numpy(dtype=int)
            minute_grid = np.arange(minutes.min(), minutes.max() + 1)
            minute_ratio = np.interp(minute_grid, minutes, spread_ratio)

            for minute_of_day, ratio in zip(minute_grid, minute_ratio):
                theta_spread = float(base_params['theta_spread'] * (ratio ** cls.spread_power))
                theta_imb = float(np.clip(
                    base_params['theta_imb'] + cls.imb_sensitivity * (ratio - 1.0),
                    cls.min_theta_imb,
                    cls.max_theta_imb,
                ))
                fitted[(arch, int(minute_of_day))] = {
                    'theta_imb': theta_imb,
                    'theta_spread': theta_spread,
                }

            details[(arch, 'base')] = base_params
            details[(arch, 'minute_profile')] = minute_profile.assign(
                smoothed_spread=spread_curve,
                spread_ratio=spread_ratio,
            )
            details[(arch, 'side_details')] = side_details

        return fitted, details

    @classmethod
    def lookup_params(cls, fitted_params, grp):
        """Use the minute-of-day schedule, with edge fallback when needed."""
        arch = grp['archetype'].iloc[0]
        minute_of_day = int(grp['minute_start'].iloc[0] // 60)
        key = (arch, minute_of_day)
        if key in fitted_params:
            return fitted_params[key]

        arch_minutes = sorted(
            k[1] for k in fitted_params
            if k[0] == arch and isinstance(k[1], (int, np.integer))
        )
        if not arch_minutes:
            return None
        if minute_of_day <= arch_minutes[0]:
            return fitted_params[(arch, arch_minutes[0])]
        if minute_of_day >= arch_minutes[-1]:
            return fitted_params[(arch, arch_minutes[-1])]

        nearest = min(arch_minutes, key=lambda m: abs(m - minute_of_day))
        return fitted_params[(arch, nearest)]


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
            'tod_bucket': grp['tod_bucket'].iloc[0],
            'minute_of_day': int(minute // 60),
        })
    return minutes


def _score_strategy_on_precomputed(strategy_cls, params, precomputed, side, signal_fn):
    """Score a single parameter set on precomputed data. Returns mean improvement."""
    strat = strategy_cls.from_grid_point(signal_fn=signal_fn, **params)
    total = 0.0
    n = len(precomputed)
    if n == 0:
        return 0.0

    for m in precomputed:
        signal = m['signal'] if side == 'buy' else 1.0 - m['signal']
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


def _fit_params_for_precomputed(strategy_cls, precomputed, archetype, signal_fn, smooth_size):
    """Fit one parameter set for an already filtered collection of minutes."""
    side_params = {}
    side_details = {}

    for side in ['buy', 'sell']:
        param_names, grid_arrays, raw_scores = grid_search(
            strategy_cls, precomputed, archetype, side, signal_fn
        )
        best_params, best_score, smoothed = smooth_and_select(
            raw_scores, grid_arrays, param_names, smooth_size
        )
        side_params[side] = best_params
        side_details[side] = (best_params, best_score, smoothed, param_names, grid_arrays)

    all_keys = list(side_params['buy'].keys())
    averaged = {
        k: (side_params['buy'][k] + side_params['sell'][k]) / 2 for k in all_keys
    }
    return averaged, side_details


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


def _fit_bucketed_strategy(strategy_cls, train_data, config, signal_fn='oi'):
    """Fit a strategy's parameters via grid search + smoothing.

    Returns
    -------
    fitted : dict — {(archetype, tod_bucket): best_params_dict}
    details : dict — {(archetype, tod_bucket, side): (best_params, best_score, smoothed_surface, param_names, grid_arrays)}
    """
    smooth_size = config.get('smooth_size', 3)
    tod_buckets = list(config.get('tod_buckets', {'mid': None}).keys())

    fitted = {}
    details = {}

    for arch in ['penny', 'wide']:
        pre_all = precompute_minute_data(
            train_data[train_data['archetype'] == arch], signal_fn
        )

        for bucket in tod_buckets:
            pre = [m for m in pre_all if m['tod_bucket'] == bucket]
            if len(pre) < 5:
                continue

            fitted[(arch, bucket)], side_details = _fit_params_for_precomputed(
                strategy_cls, pre, arch, signal_fn, smooth_size
            )
            for side, detail in side_details.items():
                details[(arch, bucket, side)] = detail

    return fitted, details


def fit_strategy(strategy_cls, train_data, config, signal_fn='oi'):
    """Dispatch to the strategy class's fitting rule."""
    return strategy_cls.fit_params(train_data, config, signal_fn)


# ---------------------------------------------------------------------------
# run_experiment: strategy + config → results
# ---------------------------------------------------------------------------

def run_experiment(strategy_cls, config, signal_fn='oi', verbose=True):
    """End-to-end: load data, fit strategy, backtest, return results.

    Parameters
    ----------
    strategy_cls : subclass of BaseStrategy
    config : dict (see utils/config.py for defaults)
    signal_fn : str or callable — which signal to use
    verbose : bool — print progress

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

    if verbose:
        print(f"Running experiment: {strategy_cls.name} (signal={signal_fn})")

    data, frames, archetypes = load_all_stocks(config)
    if verbose:
        for t in config['stocks']:
            print(f"  {t}: {archetypes[t]}")

    train_data, test_data = train_test_split(data, config)
    if verbose:
        print(f"  Train: {train_data['minute_start'].nunique()} min, "
              f"Test: {test_data['minute_start'].nunique()} min")

    if verbose:
        print("  Fitting parameters...")
    fitted, details = fit_strategy(strategy_cls, train_data, config, signal_fn)
    if verbose:
        for key, params in fitted.items():
            if not all(isinstance(v, (int, float, np.floating)) for v in params.values()):
                continue
            param_str = ', '.join(f'{k}={v:.4f}' for k, v in params.items())
            print(f"    {key}: {param_str}")

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
