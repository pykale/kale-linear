"""
kalelinear.

Learning harmonized or individualized models from multi-source/multi-view data in linear or reproducing kernel Hilbert spaces (RKHS).
"""

from importlib import import_module

__version__ = "0.1.0a1"
__author__ = "Shuo Zhou"
__credits__ = "Machine Learning Group, School of Computer Science, the University of Sheffield"

__all__ = ["embed", "predict"]


def __getattr__(name):
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
