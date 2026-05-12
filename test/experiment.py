from utils.evaluation import evaluate_both_sides
from utils.strategy import (
    OIThresholdStrategy,
    SlopeStrategy,
    TwoWindowStrategy,
    fit_slope,
    fit_strategy,
    fit_two_window,
)

from .preprocessing import load_all_stocks


def _fit_on_all_train(strategy_cls, train_data, config, signal_fn):
    """Mirror run_experiment() fitting logic without splitting train data."""
    if strategy_cls is TwoWindowStrategy:
        base_fitted, _ = fit_strategy(OIThresholdStrategy, train_data, config, signal_fn)
        return fit_two_window(train_data, config, signal_fn, base_fitted=base_fitted)
    if strategy_cls is SlopeStrategy:
        base_fitted, _ = fit_strategy(OIThresholdStrategy, train_data, config, signal_fn)
        return fit_slope(train_data, config, signal_fn, base_fitted=base_fitted)
    return fit_strategy(strategy_cls, train_data, config, signal_fn)


def run_external_test_experiment(strategy_cls, config, signal_fn='oi', verbose=True):
    """Fit on all train rows and evaluate once on all held-out test rows."""
    if verbose:
        print(f"Running held-out test experiment: {strategy_cls.name} (signal={signal_fn})")

    train_data, train_frames, archetypes = load_all_stocks(config, split='train')
    if verbose:
        for ticker in config['stocks']:
            print(f"  train {ticker}: {archetypes[ticker]}")
        print(f"  Train rows: {len(train_data):,}")

    if verbose:
        print("  Fitting parameters on full train set...")
    fitted, details = _fit_on_all_train(strategy_cls, train_data, config, signal_fn)

    if verbose:
        for arch, params in fitted.items():
            if isinstance(arch, tuple):
                continue
            param_str = ', '.join(f'{key}={value:.4f}' for key, value in params.items())
            print(f"    {arch}: {param_str}")

    test_data, test_frames, _ = load_all_stocks(config, split='test', archetypes=archetypes)
    if verbose:
        print(f"  Test rows:  {len(test_data):,}")
        print("  Backtesting on full held-out test set...")

    test_all = evaluate_both_sides(strategy_cls, test_data, fitted, signal_fn)

    return {
        'fitted': fitted,
        'details': details,
        'test_all': test_all,
        'data': train_data,
        'frames': train_frames,
        'archetypes': archetypes,
        'train_data': train_data,
        'test_data': test_data,
        'train_frames': train_frames,
        'test_frames': test_frames,
    }
