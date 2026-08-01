"""
kalelinear.

Learning harmonized or individualized models from multi-source/multi-view data in linear or reproducing kernel Hilbert spaces (RKHS).
"""

from importlib import import_module

__version__ = "0.1.0a1"

__all__ = ["transformer", "estimator", "embed", "predict"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
