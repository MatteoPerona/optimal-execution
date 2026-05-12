"""Held-out evaluation workflow.

This package keeps external test orchestration separate from the core
strategy-development pipeline in ``utils/``.
"""

from .experiment import run_external_test_experiment

__all__ = ['run_external_test_experiment']
