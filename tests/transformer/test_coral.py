import numpy as np
import pytest
from numpy import testing
from sklearn.exceptions import NotFittedError

from kalelinear.transformer import CORAL


@pytest.fixture(scope="module")
def coral_data():
    rng = np.random.default_rng(0)
    n_source, n_target, n_features = 80, 120, 3

    X_source = rng.normal(size=(n_source, n_features)) * np.array([0.5, 1.0, 2.0])
    X_target = rng.normal(size=(n_target, n_features)) * np.array([2.0, 1.0, 0.5])
    domains = np.concatenate((np.zeros(n_source, dtype=int), np.ones(n_target, dtype=int)))
    X = np.vstack((X_source, X_target))
    return X, domains, X_source, X_target


def test_coral_aligns_source_covariance_to_target(coral_data):
    X, domains, X_source, X_target = coral_data
    coral = CORAL(lambda_=1e-6)

    z = coral.fit_transform(X, covariates=domains, target_covariate=1)
    z_source = z[domains == 0]
    z_target = z[domains == 1]

    covariance_before = np.linalg.norm(np.cov(X_source, rowvar=False) - np.cov(X_target, rowvar=False))
    covariance_after = np.linalg.norm(np.cov(z_source, rowvar=False) - np.cov(z_target, rowvar=False))

    assert z_source.shape == (X_source.shape[0], X.shape[1])
    assert z_target.shape == (X_target.shape[0], X.shape[1])
    assert covariance_after < covariance_before / 100
    assert np.isfinite(z).all()


def test_coral_centers_and_aligns_rows_independently(coral_data):
    X, domains, X_source, X_target = coral_data
    coral = CORAL(lambda_=1e-6).fit(X, covariates=domains, target_covariate=1)

    expected_source = (X_source - coral.source_mean_) @ coral.alignment_
    expected_target = X_target - coral.target_mean_

    testing.assert_allclose(
        coral.transform(X_source),
        expected_source,
    )
    testing.assert_allclose(
        coral.transform(X_target, covariates=np.ones(X_target.shape[0], dtype=int)),
        expected_target,
    )

    # Row-wise domain labels must reproduce the same rows as `fit_transform`.
    permutation = np.random.default_rng(1).permutation(X.shape[0])
    z_permuted = coral.transform(X[permutation], covariates=domains[permutation])
    z_expected = coral.fit_transform(X, covariates=domains, target_covariate=1)
    testing.assert_allclose(z_permuted, z_expected[permutation])


def test_coral_fit_transform_default_lambda(coral_data):
    X, domains, _, _ = coral_data
    coral = CORAL()

    z = coral.fit_transform(X, covariates=domains, target_covariate=1)

    assert z.shape == X.shape
    assert np.isfinite(z).all()
    assert coral.source_covariance_.shape == (X.shape[1], X.shape[1])
    assert coral.target_covariance_.shape == (X.shape[1], X.shape[1])
    assert coral.alignment_.shape == (X.shape[1], X.shape[1])
    assert coral._n_features_out == X.shape[1]
    testing.assert_array_equal(
        coral.get_feature_names_out(),
        np.array([f"coral{i}" for i in range(X.shape[1])]),
    )


def test_coral_transform_requires_fit(coral_data):
    X, _, _, _ = coral_data
    with pytest.raises(NotFittedError, match="not fitted"):
        CORAL().transform(X)


def test_coral_fit_requires_covariates(coral_data):
    X, _, _, _ = coral_data
    with pytest.raises(ValueError, match="Covariates must be provided"):
        CORAL().fit(X)


def test_coral_fit_requires_both_domains(coral_data):
    X, _, X_source, _ = coral_data
    with pytest.raises(ValueError, match="both source and target"):
        CORAL().fit(X, covariates=np.zeros(X.shape[0], dtype=int))
    with pytest.raises(ValueError, match="both source and target"):
        CORAL().fit(X_source, covariates=np.zeros(X_source.shape[0], dtype=int))


def test_coral_requires_two_samples_per_domain(coral_data):
    X, _, _, _ = coral_data
    domains = np.array([0, 0, 1])
    with pytest.raises(ValueError, match="At least two target samples"):
        CORAL().fit(X[:3], covariates=domains, target_covariate=1)


def test_coral_validates_target_covariate(coral_data):
    X, domains, _, _ = coral_data
    with pytest.raises(ValueError, match="target_covariate"):
        CORAL().fit(X, covariates=domains, target_covariate=2)


def test_coral_validates_lambda(coral_data):
    X, domains, _, _ = coral_data
    with pytest.raises(ValueError, match="lambda_"):
        CORAL(lambda_=-1).fit(X, covariates=domains)


def test_coral_transform_validates_inputs(coral_data):
    X, domains, X_source, X_target = coral_data
    coral = CORAL().fit(X, covariates=domains, target_covariate=1)

    with pytest.raises(ValueError, match="features"):
        coral.transform(X[:, :2])

    with pytest.raises(ValueError, match="numeric or boolean"):
        coral.transform(X, covariates=domains.astype(str))

    with pytest.raises(ValueError, match="not seen at fit time"):
        coral.transform(X, covariates=np.full(X.shape[0], 2, dtype=int))

    with pytest.raises(ValueError, match="same number of samples"):
        coral.transform(X, covariates=np.zeros(X_target.shape[0] + 1, dtype=int))

    with pytest.raises(ValueError, match="1D array"):
        coral.transform(X_source, covariates=np.zeros((X_source.shape[0], 1, 1)))


def test_coral_accepts_boolean_domain_labels(coral_data):
    X, _, _, _ = coral_data
    domains = np.concatenate((np.zeros(X.shape[0] - 2, dtype=bool), np.ones(2, dtype=bool)))
    # Keep at least two samples per domain.
    domains[:2] = False

    coral = CORAL(lambda_=1e-6)
    z = coral.fit_transform(X, covariates=domains, target_covariate=True)

    assert coral.target_covariate_ is True
    assert z.shape == X.shape
    assert np.isfinite(z).all()
