import sys

import kalelinear
from kalelinear import embed, estimator, predict, transformer


def test_embed_module_exposes_transformers():
    assert embed.TCA is transformer.TCA
    assert embed.JDA is transformer.JDA
    assert embed.BDA is transformer.BDA
    assert embed.MIDA is transformer.MIDA
    assert embed.MPCA is transformer.MPCA
    assert embed.CIFE is transformer.CIFE
    assert embed.AJIVE is transformer.AJIVE


def test_predict_module_exposes_estimators():
    assert predict.ARSVM is estimator.ARSVM
    assert predict.ARRLS is estimator.ARRLS
    assert predict.CoIRSVM is estimator.CoIRSVM
    assert predict.CoIRLS is estimator.CoIRLS
    assert predict.GSDA is estimator.GSDA
    assert predict.LapSVM is estimator.LapSVM
    assert predict.LapRLS is estimator.LapRLS


def test_lazy_modules_are_cached_on_package():
    assert kalelinear.transformer is transformer
    assert kalelinear.estimator is estimator
    assert kalelinear.embed is embed
    assert kalelinear.predict is predict


def test_lazy_attribute_load_triggers_getattr():
    # Remove cached entries to simulate a cold attribute access through __getattr__.
    # A plain `from kalelinear import embed` bypasses __getattr__ via submodule
    # fallback, so this is the only way to actually exercise the lazy-load path.
    for name in ("embed", "predict"):
        kalelinear.__dict__.pop(name, None)
        sys.modules.pop(f"kalelinear.{name}", None)

    loaded_embed = kalelinear.embed
    assert loaded_embed is sys.modules["kalelinear.embed"]

    loaded_predict = kalelinear.predict
    assert loaded_predict is sys.modules["kalelinear.predict"]

    # Second access must return the same cached object without re-importing.
    assert kalelinear.embed is loaded_embed
    assert kalelinear.predict is loaded_predict
