# =============================================================================
# @author: Shuo Zhou, The University of Sheffield, shuo.zhou@sheffield.ac.uk
# =============================================================================
"""Correlation Alignment (CORAL)."""

from numbers import Real

import numpy as np
from sklearn.base import _fit_context, BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from kalelinear._domain import check_binary_domain_covariates, split_domain_indices


def _regularized_covariance(X, ridge):
    """Return the sample covariance of ``X`` with ridge regularization.

    The sample covariance is centered and normalized by ``n_samples - 1``,
    matching the classical CORAL implementation. ``ridge`` is added to the
    diagonal to keep the whitening step stable when the covariance is
    (close to) singular.
    """
    _, n_features = X.shape
    covariance = np.cov(X, rowvar=False)
    return covariance + ridge * np.eye(n_features, dtype=covariance.dtype)


def _symmetric_sqrt(matrix):
    """Return the symmetric positive square root of a symmetric matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 0, None)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def _symmetric_inv_sqrt(matrix):
    """Return the symmetric inverse square root via a PSD pseudo-inverse."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 0, None)
    max_eigenvalue = eigenvalues.max()
    if max_eigenvalue <= 0:
        raise ValueError(
            "The source covariance matrix has no positive eigenvalue. "
            "CORAL cannot compute a whitening transform; try a positive `lambda_`."
        )

    tolerance = 10 * np.finfo(eigenvalues.dtype).eps * max(1.0, max_eigenvalue) * eigenvalues.shape[0]
    keep = eigenvalues > tolerance
    inverse_sqrt = np.zeros_like(eigenvalues)
    inverse_sqrt[keep] = 1.0 / np.sqrt(eigenvalues[keep])
    return (eigenvectors * inverse_sqrt) @ eigenvectors.T


class CORAL(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Correlation Alignment (CORAL) for unsupervised domain adaptation.

    CORAL aligns the second-order statistics of the source and target
    feature distributions by whitening the (mean-centered) source features
    and recoloring them with the target covariance:

    .. math::

        A = (C_S + \\lambda I)^{-1/2} (C_T + \\lambda I)^{1/2},

    where :math:`C_S` and :math:`C_T` are the source and target sample
    covariance matrices. Samples are centered by their domain mean during
    :meth:`fit_transform`, so both domains are embedded in a common
    mean-centered feature space. Source samples are transformed with the
    learned alignment :math:`A`; target samples are only centered.

    ``covariates`` represent binary domain labels of length ``n_samples``.
    They must contain both source and target domains during :meth:`fit`.
    ``target_covariate`` selects which label is treated as the target domain.

    Parameters
    ----------
    lambda_ : float, default=1e-5
        Regularization added to the diagonal of the source and target
        covariance matrices before computing the alignment. Use a larger
        value when the covariance estimates are unstable (e.g. very few
        samples or near-constant features).

    Attributes
    ----------
    source_mean_ : ndarray of shape (n_features,)
        Mean of the source training samples.
    target_mean_ : ndarray of shape (n_features,)
        Mean of the target training samples.
    source_covariance_ : ndarray of shape (n_features, n_features)
        Regularized source covariance matrix.
    target_covariance_ : ndarray of shape (n_features, n_features)
        Regularized target covariance matrix.
    alignment_ : ndarray of shape (n_features, n_features)
        Linear whitening/recoloring transform applied to source features.
    target_covariate_ : object
        Domain label treated as the target domain.

    References
    ----------
    Sun, B., Feng, J. and Saenko, K., 2016. Return of Frustratingly Easy
    Domain Adaptation. In *Proceedings of the AAAI Conference on Artificial
    Intelligence*.
    """

    _parameter_constraints: dict = {
        "lambda_": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(self, lambda_=1e-5):
        self.lambda_ = lambda_

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, covariates=None, target_covariate=None):
        """Fit CORAL on source and target samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Source and target samples stacked along the first axis.
        y : array-like, default=None
            Ignored. Present for scikit-learn API consistency. CORAL is an
            unsupervised domain adaptation method.
        covariates : array-like of shape (n_samples,), default=None
            Binary domain label for every sample. Must contain both source
            and target domain values.
        target_covariate : object, default=None
            Domain label treated as the target domain. Defaults to the last
            unique covariate value, as in the other KaleLinear adapters.

        Returns
        -------
        self : CORAL
            Fitted transformer.
        """
        X = validate_data(self, X, dtype=[np.float64, np.float32])
        n_samples, n_features = X.shape

        if covariates is None:
            raise ValueError("Covariates must be provided for CORAL during `fit`.")

        covariates, unique_covariates = check_binary_domain_covariates(
            covariates,
            n_samples,
            require_numeric=True,
            error_prefix=f"Covariates for {self.__class__.__name__}",
            both_domains_message=(
                f"Covariates for {self.__class__.__name__} must contain both source and target domain values."
            ),
        )
        split = split_domain_indices(covariates, target_covariate)

        X_source = X[split.source_idx]
        X_target = X[split.target_idx]
        if X_source.shape[0] < 2:
            raise ValueError("At least two source samples are required to estimate the source covariance.")
        if X_target.shape[0] < 2:
            raise ValueError("At least two target samples are required to estimate the target covariance.")

        self.source_mean_ = X_source.mean(axis=0)
        self.target_mean_ = X_target.mean(axis=0)
        self.source_covariance_ = _regularized_covariance(X_source, self.lambda_)
        self.target_covariance_ = _regularized_covariance(X_target, self.lambda_)
        self.alignment_ = _symmetric_inv_sqrt(self.source_covariance_) @ _symmetric_sqrt(self.target_covariance_)
        self.target_covariate_ = split.target_covariate
        self.domain_values_ = unique_covariates
        self._n_features_out = n_features
        return self

    def transform(self, X, covariates=None):
        """Align new samples to the learned target feature space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.
        covariates : array-like of shape (n_samples,), default=None
            Binary domain labels for the new samples. Source rows are
            whitened with the source covariance and recolored with the target
            covariance; target rows are centered by the fitted target mean.
            When ``None``, all rows are treated as source samples.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features)
            Transformed samples. Feature dimensionality is preserved.
        """
        check_is_fitted(self, "alignment_")
        X = validate_data(self, X, dtype=[np.float64, np.float32], reset=False)

        if covariates is None:
            return (X - self.source_mean_) @ self.alignment_

        covariates = np.asarray(covariates)
        if covariates.ndim == 2 and covariates.shape[1] == 1:
            covariates = covariates.reshape(-1)
        if covariates.ndim != 1:
            raise ValueError(f"Covariates for {self.__class__.__name__} must be a 1D array of domain labels.")
        if covariates.shape[0] != X.shape[0]:
            raise ValueError("Covariates and X must have the same number of samples.")
        if not (np.issubdtype(covariates.dtype, np.number) or np.issubdtype(covariates.dtype, np.bool_)):
            raise ValueError(f"Covariates for {self.__class__.__name__} should be numeric or boolean domain labels.")

        unknown_values = np.setdiff1d(np.unique(covariates), self.domain_values_)
        if unknown_values.size:
            raise ValueError(
                f"Covariates for {self.__class__.__name__} contain domain values "
                f"not seen at fit time: {unknown_values.tolist()}."
            )

        source_mask = covariates != self.target_covariate_
        X_new = (X - self.target_mean_).copy()
        if np.any(source_mask):
            X_new[source_mask] = (X[source_mask] - self.source_mean_) @ self.alignment_
        return X_new

    def fit_transform(self, X, y=None, covariates=None, target_covariate=None):
        """Fit CORAL and transform the source and target samples.

        Parameters are the same as for :meth:`fit`. Domain labels are
        forwarded to :meth:`transform`, so source samples are aligned and
        target samples are centered in the returned array.
        """
        self.fit(X, y=y, covariates=covariates, target_covariate=target_covariate)
        return self.transform(X, covariates=covariates)
