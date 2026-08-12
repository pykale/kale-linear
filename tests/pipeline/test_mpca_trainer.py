import numpy as np
import pytest
from numpy import testing
from sklearn.metrics import accuracy_score, roc_auc_score

from kalelinear.pipeline.mpca_trainer import MPCATrainer

CLASSIFIERS = ["svc", "linear_svc", "lr"]
PARAMS = [
    {"classifier_params": "auto", "mpca_params": None, "n_features": None, "search_params": None},
    {
        "classifier_params": {"C": 1},
        "mpca_params": {"explained_variance_ratio": 0.9, "vectorize": True},
        "n_features": 100,
        "search_params": {"cv": 3},
    },
]


@pytest.mark.parametrize("classifier", CLASSIFIERS)
@pytest.mark.parametrize("params", PARAMS)
def test_mpca_trainer(classifier, params, gait):
    X = gait["fea3D"].transpose((3, 0, 1, 2))
    X = X[:20, :]
    y = gait["gnd"][:20].reshape(-1)
    trainer = MPCATrainer(classifier=classifier, **params)
    trainer.fit(X, y)
    y_pred = trainer.predict(X)
    testing.assert_equal(np.unique(y), np.unique(y_pred))
    assert accuracy_score(y, y_pred) >= 0.8

    if classifier == "linear_svc":
        with pytest.raises(Exception):
            y_proba = trainer.predict_proba(X)
    else:
        y_proba = trainer.predict_proba(X)
        assert np.max(y_proba) <= 1.0
        assert np.min(y_proba) >= 0.0
        y_ = np.zeros(y.shape)
        y_[np.where(y == 1)] = 1
        assert roc_auc_score(y_, y_proba[:, 0]) >= 0.8

    y_dec_score = trainer.decision_function(X)
    assert roc_auc_score(y, y_dec_score) >= 0.8

    if classifier == "svc" and trainer.clf.kernel == "rbf":
        with pytest.raises(Exception):
            trainer.mpca.inverse_transform(trainer.clf.coef_)
    else:
        # interpret utilities (select_top_weight/plot_weights) are not ported
        # to kalelinear yet, so only check the inverse-transform path here.
        weights = trainer.mpca.inverse_transform(trainer.clf.coef_) - trainer.mpca.mean_
        testing.assert_equal(weights.shape[1:], X.shape[1:])


def test_invalid_init():
    with pytest.raises(Exception):
        MPCATrainer(classifier="Ridge")
    with pytest.raises(Exception):
        MPCATrainer(classifier_params=False)
