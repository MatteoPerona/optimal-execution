"""Wrap v9_final strategies so they can run through the main eval pipeline."""

import numpy as np

from .strategy import BaseStrategy
from v9_final.strategies import (
    AdaptiveDeadline as _AdaptiveDeadline,
    Ensemble as _Ensemble,
    Microprice as _Microprice,
    OFIContrarian as _OFIContrarian,
    PARAM_GRIDS,
    SpreadQuantile as _SpreadQuantile,
)


class _AdapterBase(BaseStrategy):
    """Translate the main strategy interface into the v9 execute() API."""

    _COMBOS = []

    def __init__(self, params, signal_fn="oi"):
        super().__init__(params, signal_fn)
        idx = int(round(params["combo_idx"]))
        idx = max(0, min(idx, len(self._COMBOS) - 1))
        combo = self._COMBOS[idx]
        self._inner = self._build_inner(combo)

    def _build_inner(self, combo):
        raise NotImplementedError

    def execute_minute(self, grp, side):
        if "spread" in grp.columns and "Spread" not in grp.columns:
            grp = grp.rename(columns={"spread": "Spread"})

        return self._inner.execute(grp, side.upper(), {})

    def decide(self, signal, spread, side):
        raise NotImplementedError("Adapter strategies override execute_minute directly.")

    @classmethod
    def param_grid(cls, archetype):
        return {"combo_idx": np.arange(len(cls._COMBOS), dtype=float)}

    @classmethod
    def from_grid_point(cls, signal_fn="oi", **kwargs):
        return cls(params=kwargs, signal_fn=signal_fn)


class SpreadQuantileAdapter(_AdapterBase):
    name = "SpreadQuantile"
    _COMBOS = PARAM_GRIDS["SpreadQuantile"]

    def _build_inner(self, combo):
        return _SpreadQuantile(**combo)


class OFIContrarianAdapter(_AdapterBase):
    name = "OFIContrarian"
    _COMBOS = PARAM_GRIDS["OFIContrarian"]

    def _build_inner(self, combo):
        return _OFIContrarian(**combo)


class MicropriceAdapter(_AdapterBase):
    name = "Microprice"
    _COMBOS = PARAM_GRIDS["Microprice"]

    def _build_inner(self, combo):
        return _Microprice(**combo)


class AdaptiveDeadlineAdapter(_AdapterBase):
    name = "AdaptiveDeadline"
    _COMBOS = PARAM_GRIDS["AdaptiveDeadline"]

    def _build_inner(self, combo):
        return _AdaptiveDeadline(**combo)


class EnsembleAdapter(_AdapterBase):
    name = "Ensemble"
    _COMBOS = PARAM_GRIDS["Ensemble"]

    def _build_inner(self, combo):
        return _Ensemble(**combo)


def run_experiment_v9(strategy_cls, config, verbose=True):
    """Fit v9 fixed parameter combos and backtest via the main evaluation path."""

    from .evaluation import evaluate_both_sides
    from .preprocessing import load_all_stocks, train_test_split

    if verbose:
        print(f"Running experiment: {strategy_cls.name}")

    data, frames, archetypes = load_all_stocks(config)
    if verbose:
        for ticker in config["stocks"]:
            print(f"  {ticker}: {archetypes[ticker]}")

    train_data, test_data = train_test_split(data, config)
    if verbose:
        print(
            f"  Train: {train_data['minute_start'].nunique()} min, "
            f"Test:  {test_data['minute_start'].nunique()} min"
        )
        print("  Fitting parameters (iterating over fixed combos)...")

    fitted = {}
    n_combos = len(strategy_cls._COMBOS)

    for arch in ["penny", "wide"]:
        arch_train = train_data[train_data["archetype"] == arch]
        best_score = -np.inf
        best_params = {"combo_idx": 0.0}

        for combo_idx in range(n_combos):
            params = {"combo_idx": float(combo_idx)}
            strat = strategy_cls(params=params, signal_fn="oi")

            total = 0.0
            n = 0
            for (_, _), grp in arch_train.groupby(["ticker", "minute_start"]):
                if len(grp) < 5:
                    continue
                for side in ["buy", "sell"]:
                    price = strat.execute_minute(grp, side)
                    twap = grp["twap_ask"].iloc[0] if side == "buy" else grp["twap_bid"].iloc[0]
                    total += (twap - price) if side == "buy" else (price - twap)
                    n += 1

            score = total / n if n > 0 else 0.0
            if score > best_score:
                best_score = score
                best_params = params

        fitted[arch] = best_params
        if verbose:
            idx = int(round(best_params["combo_idx"]))
            combo = strategy_cls._COMBOS[idx]
            print(f"    {arch}: combo {idx} {combo}  score={best_score:.6f}")

    if verbose:
        print("  Backtesting...")

    test_all = evaluate_both_sides(strategy_cls, test_data, fitted, signal_fn="oi")

    return {
        "fitted": fitted,
        "details": {},
        "test_all": test_all,
        "data": data,
        "frames": frames,
        "archetypes": archetypes,
        "train_data": train_data,
        "test_data": test_data,
    }
