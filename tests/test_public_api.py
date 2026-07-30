import kalelinear
from kalelinear import embed, estimator, predict, transformer


def test_embed_module_exposes_transformers():
    assert embed.TCA is transformer.TCA
    assert embed.JDA is transformer.JDA
    assert embed.BDA is transformer.BDA
    assert embed.MIDA is transformer.MIDA
    assert embed.MPCA is transformer.MPCA


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
