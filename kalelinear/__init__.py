"""
kalelinear.

Non-deep machine learning that learns transferable, shared, or group-specific models from data across multiple sources, groups, blocks, or views.
"""

from importlib import import_module

__version__ = "0.1.0b1"

__all__ = ["transformer", "estimator", "embed", "predict"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
