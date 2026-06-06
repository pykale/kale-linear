"""Compatibility wrapper for plotting utilities.

Prefer importing from ``utils.plotting`` in new code.
"""

from .plotting import load_coef_plot_corr, load_weight_plot_corr, savefig

__all__ = ["load_coef_plot_corr", "load_weight_plot_corr", "savefig"]
