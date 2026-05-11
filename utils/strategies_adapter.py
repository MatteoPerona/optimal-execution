"""strategies_adapter.py
=======================
Wraps every strategy from the teammate's repo (strategies.py) so they
can be evaluated using the main-repo evaluation pipeline and compared
against the baseline via compare_strategies().

Use run_experiment_v9() (defined here) instead of run_experiment() for
all v9_final strategies.  Everything else (evaluate_both_sides,
compare_strategies, print_results, plot_results) works unchanged.

Usage
-----
    from utils.strategies_adapter import (
        run_experiment_v9,
        EnsembleAdapter,
        SpreadQuantileAdapter,
        OFIContrarianAdapter,
        MicropriceAdapter,
        AdaptiveDeadlineAdapter,
    )
    from utils.evaluation import compare_strategies

    ensemble_results = run_experiment_v9(EnsembleAdapter, config)

    compare_strategies([
        ('OI Threshold', oi_results),          # from run_experiment()
        ('Ensemble',     ensemble_results),     # from run_experiment_v9()
    ])

Why run_experiment_v9() and not run_experiment()?
-------------------------------------------------
The main-repo grid search calls strat.decide(signal, spread, side),
which only receives pre-computed 1-D arrays.  The teammate's strategies
need the full minute DataFrame (BidSize_1, AskSize_1, etc.) to compute
OFI and microprice internally.  run_experiment_v9() replaces only the
fitting step with one that calls execute_minute() — which receives the
full DataFrame — so no information is lost.  The backtesting step is
identical to the main pipeline (evaluate_both_sides).
"""

import numpy as np

# Main-repo base class
from .strategy import BaseStrategy

# Teammate's strategy classes and fixed param combos.
# v9_final is a sibling package to utils (both sit at the repo root),
# so this must be an absolute import, not a relative one.
from v9_final.strategies import (
    SpreadQuantile   as _SpreadQuantile,
    OFIContrarian    as _OFIContrarian,
    Microprice       as _Microprice,
    AdaptiveDeadline as _AdaptiveDeadline,
    Ensemble         as _Ensemble,
    PARAM_GRIDS,
)


# ---------------------------------------------------------------------------
# Shared adapter base
# ---------------------------------------------------------------------------

class _AdapterBase(BaseStrategy):
    """Mixin that handles the three interface translations:

    1. Column name: renames 'spread' → 'Spread' for the teammate's code.
    2. Side casing: converts 'buy'/'sell' → 'BUY'/'SELL'.
    3. Param encoding: stores a single integer 'combo_idx' that indexes
       into the class-level _COMBOS list, so the main-repo grid search
       can iterate over it with np.arange.

    Subclasses must define:
        _COMBOS : list of dicts   — copied from PARAM_GRIDS
        _build_inner(combo)       — construct the teammate strategy instance
    """

    _COMBOS: list = []   # override in each subclass

    def __init__(self, params, signal_fn='oi'):
        super().__init__(params, signal_fn)
        idx = int(round(params['combo_idx']))
        idx = max(0, min(idx, len(self._COMBOS) - 1))   # clamp to valid range
        combo = self._COMBOS[idx]
        self._inner = self._build_inner(combo)

    def _build_inner(self, combo):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Interface translation — overrides execute_minute so decide() is
    # never called (it's left as NotImplementedError on purpose).
    # ------------------------------------------------------------------

    def execute_minute(self, grp, side):
        """Translate main-repo call signature → teammate's execute()."""

        # 1. Column name: main repo uses lowercase 'spread';
        #    teammate's code uses capital 'Spread'.
        if 'spread' in grp.columns and 'Spread' not in grp.columns:
            grp = grp.rename(columns={'spread': 'Spread'})

        # 2. Side casing
        side_upper = side.upper()   # 'buy' → 'BUY', 'sell' → 'SELL'

        # 3. ctx: pass empty dict → each strategy falls back gracefully.
        #    See the note in the module docstring about rolling context.
        ctx = {}

        return self._inner.execute(grp, side_upper, ctx)

    def decide(self, signal, spread, side):
        # Never called (execute_minute is overridden), but required by ABC.
        raise NotImplementedError(
            "Adapter strategies override execute_minute directly."
        )

    # ------------------------------------------------------------------
    # Param grid: one integer axis → indexes into _COMBOS list
    # ------------------------------------------------------------------

    @classmethod
    def param_grid(cls, archetype):
        """Return a single-axis grid over combo indices.

        The main-repo grid search does a Cartesian product over this.
        With one axis of length N it simply tries each of the N combos.
        """
        return {'combo_idx': np.arange(len(cls._COMBOS), dtype=float)}

    @classmethod
    def from_grid_point(cls, signal_fn='oi', **kwargs):
        return cls(params=kwargs, signal_fn=signal_fn)


# ---------------------------------------------------------------------------
# Concrete adapters — one per teammate strategy
# ---------------------------------------------------------------------------

class SpreadQuantileAdapter(_AdapterBase):
    """Wraps strategies.SpreadQuantile."""

    name = 'SpreadQuantile'
    _COMBOS = PARAM_GRIDS['SpreadQuantile']

    def _build_inner(self, combo):
        return _SpreadQuantile(**combo)


class OFIContrarianAdapter(_AdapterBase):
    """Wraps strategies.OFIContrarian."""

    name = 'OFIContrarian'
    _COMBOS = PARAM_GRIDS['OFIContrarian']

    def _build_inner(self, combo):
        return _OFIContrarian(**combo)


class MicropriceAdapter(_AdapterBase):
    """Wraps strategies.Microprice."""

    name = 'Microprice'
    _COMBOS = PARAM_GRIDS['Microprice']

    def _build_inner(self, combo):
        return _Microprice(**combo)


class AdaptiveDeadlineAdapter(_AdapterBase):
    """Wraps strategies.AdaptiveDeadline."""

    name = 'AdaptiveDeadline'
    _COMBOS = PARAM_GRIDS['AdaptiveDeadline']

    def _build_inner(self, combo):
        return _AdaptiveDeadline(**combo)


class EnsembleAdapter(_AdapterBase):
    """Wraps strategies.Ensemble."""

    name = 'Ensemble'
    _COMBOS = PARAM_GRIDS['Ensemble']

    def _build_inner(self, combo):
        return _Ensemble(**combo)


# ---------------------------------------------------------------------------
# run_experiment_v9: the right entry point for all adapter strategies
# ---------------------------------------------------------------------------

def run_experiment_v9(strategy_cls, config, verbose=True):
    """End-to-end experiment runner for v9_final adapter strategies.

    Mirrors run_experiment() from strategy.py but replaces the grid-search
    fitting step with one that calls execute_minute() on the full minute
    DataFrame.  This is necessary because the teammate's strategies compute
    OFI and microprice internally from raw bid/ask size columns, which are
    not available in the pre-computed signal/spread arrays that decide()
    receives in the main-repo grid search.

    The backtesting step is identical to the main pipeline
    (evaluate_both_sides), so results are directly comparable.

    Parameters
    ----------
    strategy_cls : subclass of _AdapterBase
    config       : dict  (same DEFAULT_CONFIG used for run_experiment)
    verbose      : bool

    Returns
    -------
    dict with the same keys as run_experiment():
        'fitted', 'test_all', 'data', 'frames', 'archetypes',
        'train_data', 'test_data'
        ('details' is empty — no smoothed surface for fixed-combo search)
    """
    from .preprocessing import load_all_stocks, train_test_split
    from .evaluation import evaluate_both_sides

    if verbose:
        print(f"Running experiment: {strategy_cls.name}")

    data, frames, archetypes = load_all_stocks(config)
    if verbose:
        for t in config['stocks']:
            print(f"  {t}: {archetypes[t]}")

    train_data, test_data = train_test_split(data, config)
    if verbose:
        print(f"  Train: {train_data['minute_start'].nunique()} min, "
              f"Test:  {test_data['minute_start'].nunique()} min")

    # ------------------------------------------------------------------
    # Fitting: iterate over fixed combos, score each via execute_minute()
    # ------------------------------------------------------------------
    if verbose:
        print("  Fitting parameters (iterating over fixed combos)...")

    fitted = {}
    n_combos = len(strategy_cls._COMBOS)

    for arch in ['penny', 'wide']:
        arch_train = train_data[train_data['archetype'] == arch]
        best_score  = -np.inf
        best_params = {'combo_idx': 0.0}

        for combo_idx in range(n_combos):
            params = {'combo_idx': float(combo_idx)}
            strat  = strategy_cls(params=params, signal_fn='oi')

            total, n = 0.0, 0
            for (_, _), grp in arch_train.groupby(['ticker', 'minute_start']):
                if len(grp) < 5:
                    continue
                for side in ['buy', 'sell']:
                    price = strat.execute_minute(grp, side)
                    twap  = (grp['twap_ask'].iloc[0] if side == 'buy'
                             else grp['twap_bid'].iloc[0])
                    total += (twap - price) if side == 'buy' else (price - twap)
                    n     += 1

            score = total / n if n > 0 else 0.0
            if score > best_score:
                best_score  = score
                best_params = params

        fitted[arch] = best_params
        if verbose:
            idx   = int(round(best_params['combo_idx']))
            combo = strategy_cls._COMBOS[idx]
            print(f"    {arch}: combo {idx} {combo}  score={best_score:.6f}")

    # ------------------------------------------------------------------
    # Backtesting: identical to the main pipeline
    # ------------------------------------------------------------------
    if verbose:
        print("  Backtesting...")

    test_all = evaluate_both_sides(strategy_cls, test_data, fitted, signal_fn='oi')

    return {
        'fitted':     fitted,
        'details':    {},          # no smoothed surface for fixed-combo search
        'test_all':   test_all,
        'data':       data,
        'frames':     frames,
        'archetypes': archetypes,
        'train_data': train_data,
        'test_data':  test_data,
    }