from kalelinear import embed, predict


def test_embed_module_exposes_transformers():
    assert embed.TCA.__name__ == "TCA"
    assert embed.JDA.__name__ == "JDA"
    assert embed.BDA.__name__ == "BDA"
    assert embed.MIDA.__name__ == "MIDA"
    assert embed.MPCA.__name__ == "MPCA"


def test_predict_module_exposes_estimators():
    assert predict.ARSVM.__name__ == "ARSVM"
    assert predict.ARRLS.__name__ == "ARRLS"
    assert predict.CoIRSVM.__name__ == "CoIRSVM"
    assert predict.CoIRLS.__name__ == "CoIRLS"
    assert predict.GSDA.__name__ == "GSDA"
    assert predict.LapSVM.__name__ == "LapSVM"
    assert predict.LapRLS.__name__ == "LapRLS"
