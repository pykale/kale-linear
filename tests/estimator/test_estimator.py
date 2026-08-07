import numpy as np
import pytest
from sklearn.metrics import accuracy_score

import kalelinear.estimator as estimator


@pytest.fixture
def office_test_data():
    X = np.array(
        [
            [-2.2, -1.9],
            [-1.9, -2.1],
            [-1.7, -1.8],
            [1.8, 2.1],
            [2.0, 1.9],
            [2.2, 2.0],
            [-1.4, -1.2],
            [-1.1, -1.3],
            [1.3, 1.1],
            [1.5, 1.2],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    z = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    covariate_mat = np.eye(2)[z]
    return X, y, z, covariate_mat


def _split_source_target(office_test_data, target_domain=0):
    X, y, z, covariate_mat = office_test_data
    tgt_idx = np.where(z == target_domain)
    src_idx = np.where(z != target_domain)
    X_train = np.concatenate((X[src_idx], X[tgt_idx]))
    c_train = np.concatenate((covariate_mat[src_idx], covariate_mat[tgt_idx]))
    y_train = y[src_idx]
    return X, y, tgt_idx, src_idx, X_train, c_train, y_train


def test_coir_svm_solvers_fit_consistently(office_test_data):
    X, y, tgt_idx, src_idx, X_train, c_train, y_train = _split_source_target(office_test_data)

    clf1 = estimator.CoIRSVM()
    clf2 = estimator.CoIRSVM(solver="cvxopt")

    clf1.fit(X_train, y_train, c_train)
    clf2.fit(X_train, y_train, c_train)

    for clf in (clf1, clf2):
        y_pred = clf.predict(X[tgt_idx])
        assert clf.coef_.shape[0] == X_train.shape[0]
        assert np.isfinite(clf.coef_).all()
        assert set(np.unique(y_pred)).issubset(set(np.unique(y)))


def test_coir_svm_none_covariates_matches_zero_covariates(office_test_data):
    X, _, tgt_idx, _, X_train, _, y_train = _split_source_target(office_test_data)
    zero_covariates = np.zeros((X_train.shape[0], 1))

    clf_none = estimator.CoIRSVM().fit(X_train, y_train, covariates=None)
    clf_zero = estimator.CoIRSVM().fit(X_train, y_train, covariates=zero_covariates)

    dec_none = clf_none.decision_function(X[tgt_idx])
    dec_zero = clf_zero.decision_function(X[tgt_idx])
    pred_none = clf_none.predict(X[tgt_idx])
    pred_zero = clf_zero.predict(X[tgt_idx])

    assert np.allclose(dec_none, dec_zero)
    assert np.array_equal(pred_none, pred_zero)


def test_coir_ls_none_covariates_matches_zero_covariates(office_test_data):
    X, _, tgt_idx, _, X_train, _, y_train = _split_source_target(office_test_data)
    zero_covariates = np.zeros((X_train.shape[0], 1))

    clf_none = estimator.CoIRLS().fit(X_train, y_train, covariates=None)
    clf_zero = estimator.CoIRLS().fit(X_train, y_train, covariates=zero_covariates)

    dec_none = clf_none.decision_function(X[tgt_idx])
    dec_zero = clf_zero.decision_function(X[tgt_idx])
    pred_none = clf_none.predict(X[tgt_idx])
    pred_zero = clf_zero.predict(X[tgt_idx])

    assert np.allclose(dec_none, dec_zero)
    assert np.array_equal(pred_none, pred_zero)


@pytest.mark.parametrize("estimator_cls", [estimator.CoIRSVM, estimator.CoIRLS])
def test_coir_estimators_predict_labels(estimator_cls, office_test_data):
    X, y, tgt_idx, src_idx, X_train, c_train, y_train = _split_source_target(office_test_data)

    clf = estimator_cls()
    clf.fit(X_train, y_train, c_train)

    decision = clf.decision_function(X[tgt_idx])
    y_pred = clf.predict(X[tgt_idx])
    acc = accuracy_score(y[tgt_idx], y_pred)

    assert decision.shape[0] == len(tgt_idx[0])
    assert y_pred.shape == y[tgt_idx].shape
    assert 0 <= acc <= 1


@pytest.mark.parametrize("estimator_cls", [estimator.CoIRSVM, estimator.CoIRLS])
def test_coir_covariate_encoder_accepts_string_covariates(estimator_cls, office_test_data):
    X, y, z, _ = office_test_data
    tgt_idx = np.where(z == 0)
    src_idx = np.where(z != 0)
    X_train = np.concatenate((X[src_idx], X[tgt_idx]))
    z_train = np.concatenate((z[src_idx], z[tgt_idx]))
    y_train = y[src_idx]
    string_covariates = np.where(z_train == 0, "target", "source")
    numeric_covariates = np.eye(2)[z_train]

    string_clf = estimator_cls(covariate_encoder="onehot").fit(X_train, y_train, string_covariates)
    numeric_clf = estimator_cls().fit(X_train, y_train, numeric_covariates)

    assert string_clf.covariate_encoder_ is not None
    assert np.allclose(string_clf.decision_function(X[tgt_idx]), numeric_clf.decision_function(X[tgt_idx]))


@pytest.mark.parametrize("estimator_cls", [estimator.ARSVM, estimator.ARRLS])
def test_artl_estimators_predict_labels(estimator_cls, office_test_data):
    X, y, z, _ = office_test_data
    tgt_idx = np.where(z == 0)
    src_idx = np.where(z != 0)

    clf = estimator_cls()
    clf.fit(X, y[src_idx], covariates=z, target_covariate=0)

    decision = clf.decision_function(X[tgt_idx])
    y_pred = clf.predict(X[tgt_idx])
    acc = accuracy_score(y[tgt_idx], y_pred)

    assert decision.shape[0] == len(tgt_idx[0])
    assert y_pred.shape == y[tgt_idx].shape
    assert 0 <= acc <= 1


@pytest.mark.parametrize("estimator_cls", [estimator.ARSVM, estimator.ARRLS])
def test_artl_covariate_api_accepts_source_only_or_full_labels(estimator_cls, office_test_data):
    X, y, z, _ = office_test_data
    tgt_idx = np.where(z == 0)
    src_idx = np.where(z != 0)

    covariate = estimator_cls().fit(X, y[src_idx], covariates=z, target_covariate=0)
    full_y = y.copy()
    full_y[tgt_idx] = -1
    covariate_full_y = estimator_cls().fit(X, full_y, covariates=z, target_covariate=0, unlabeled_value=-1)

    assert np.allclose(covariate.decision_function(X[tgt_idx]), covariate_full_y.decision_function(X[tgt_idx]))
    assert np.array_equal(covariate.predict(X[tgt_idx]), covariate_full_y.predict(X[tgt_idx]))


@pytest.mark.parametrize("estimator_cls", [estimator.ARSVM, estimator.ARRLS])
def test_artl_rejects_legacy_target_keyword(estimator_cls, office_test_data):
    X, y, z, _ = office_test_data
    tgt_idx = np.where(z == 0)
    src_idx = np.where(z != 0)

    with pytest.raises(TypeError, match="unexpected keyword argument 'Xt'"):
        estimator_cls().fit(X[src_idx], y[src_idx], Xt=X[tgt_idx])


@pytest.mark.parametrize("estimator_cls", [estimator.ARSVM, estimator.ARRLS])
def test_artl_covariate_fit_predict_returns_target_labels(estimator_cls, office_test_data):
    X, y, z, _ = office_test_data
    tgt_idx = np.where(z == 0)
    src_idx = np.where(z != 0)

    y_pred = estimator_cls().fit_predict(X, y[src_idx], covariates=z, target_covariate=0)

    assert y_pred.shape == y[tgt_idx].shape
    assert set(np.unique(y_pred)).issubset(set(np.unique(y)))


@pytest.mark.parametrize("estimator_cls", [estimator.LapSVM, estimator.LapRLS])
def test_manifold_estimators_predict_labels(estimator_cls, office_test_data):
    X, y, tgt_idx, src_idx, X_train, _, y_train = _split_source_target(office_test_data)

    clf = estimator_cls()
    clf.fit(X_train, y_train)

    decision = clf.decision_function(X[tgt_idx])
    y_pred = clf.predict(X[tgt_idx])
    acc = accuracy_score(y[tgt_idx], y_pred)

    assert decision.shape[0] == len(tgt_idx[0])
    assert y_pred.shape == y[tgt_idx].shape
    assert 0 <= acc <= 1


def test_gsda_fit_predicts_target_labels(office_test_data):
    X, y, z, covariate_mat = office_test_data
    target_idx = np.where(z == 0)[0]

    clf = estimator.GSDA(max_iter=25, random_state=0)
    clf.fit(X, y[target_idx], covariate_mat, target_idx=target_idx)

    y_proba = clf.predict_proba(X[target_idx])
    y_pred = clf.predict(X[target_idx])
    params = clf.get_fitted_params()

    assert clf.coef_.shape == (X.shape[1],)
    assert params["coef"].shape == (X.shape[1],)
    assert np.isfinite(clf.coef_).all()
    assert np.isfinite(clf.intercept_)
    assert y_proba.shape == y[target_idx].shape
    assert y_pred.shape == y[target_idx].shape
    assert np.all((0 <= y_proba) & (y_proba <= 1))
    assert set(np.unique(y_pred)).issubset({0.0, 1.0})
    assert len(clf.losses["time"]) == 1
    assert len(clf.losses["ovr"]) > 0


def test_gsda_covariate_encoder_accepts_string_groups(office_test_data):
    X, y, z, _ = office_test_data
    target_idx = np.where(z == 0)[0]
    string_groups = np.where(z == 0, "target", "source")

    clf = estimator.GSDA(max_iter=25, random_state=0, covariate_encoder="onehot")
    clf.fit(X, y[target_idx], string_groups, target_idx=target_idx)

    y_pred = clf.predict(X[target_idx])

    assert clf.covariate_encoder_ is not None
    assert y_pred.shape == y[target_idx].shape
