"""Default experiment configuration.

Change any value here or pass overrides to run_experiment().
"""

DEFAULT_CONFIG = {
    # Data
    'stocks': ['INTC', 'MSFT', 'AMZN', 'GOOG'],
    'data_dir': 'data',

    # Archetype detection
    'penny_spread_cutoff': 0.02,  # dollars

    # Train/test split
    'train_frac': 0.7,

    # Grid search ranges
    'imb_grid': (0.52, 0.95, 30),        # (start, stop, num_points)
    'spread_grid_penny': (0.005, 0.03, 20),
    'spread_grid_wide': (0.05, 0.50, 20),
    'smooth_size': 3,

    # Signal analysis
    'horizons': (10, 20, 30, 60),
    'n_bins': 20,
}
