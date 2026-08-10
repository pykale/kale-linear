# =============================================================================
# Authors: Shuo Zhou, shuo.zhou@sheffield.ac.uk
#          Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
#          Lalu Muhammad Riza Rizky, l.m.rizky@sheffield.ac.uk
# =============================================================================

"""Tensor and kernel domain adaptation transformers.

This module provides the MPCA transformer and associated utilities.
"""
import logging
import warnings

import numpy as np
from numpy.linalg import eigvalsh
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from tensorly.base import fold, unfold
from tensorly.tenalg import multi_mode_dot

_CHUNK_ELEMS = 4000000  # target number of float64 elements per processed chunk (~32 MB)


def _chunk_size(n_samples, elems_per_sample, chunk_elems=_CHUNK_ELEMS):
    """Return the largest chunk keeping per-chunk element count under ``chunk_elems``.

    Parameters
    ----------
    n_samples : int
        Number of samples in the data.
    elems_per_sample : int
        Number of array elements contributed by a single sample (e.g. the
        number of features of the input or unfolded tensor).
    chunk_elems : int, default=4000000
        Memory budget in number of elements for one processed chunk.

    Returns
    -------
    chunk_size : int
        Number of samples to process at a time.
    """
    if elems_per_sample <= 0:
        return n_samples
    return min(n_samples, max(1, chunk_elems // elems_per_sample))


def _check_n_dim(X, n_dims):
    """Validate the number of dimensions.

    Parameters
    ----------
    X : ndarray of shape (n_samples, I_1, ..., I_N)
        Input tensor data.
    n_dims : int
        Expected number of dimensions.

    Raises
    ------
    ValueError
        If ``X.ndim`` does not match ``n_dims``.
    """
    if not X.ndim == n_dims:
        error_msg = "The expected number of dimensions is %s but it is %s for given data" % (n_dims, X.ndim)
        logging.error(error_msg)
        raise ValueError(error_msg)


def _check_shape(X, shape_):
    """Validate per-sample tensor shape.

    Parameters
    ----------
    X : ndarray of shape (n_samples, I_1, ..., I_N)
        Input tensor data.
    shape_ : tuple
        Expected per-sample shape ``(I_1, ..., I_N)``.

    Raises
    ------
    ValueError
        If the sample shape of ``X`` does not match ``shape_``.
    """
    if not X.shape[1:] == shape_:
        error_msg = "The expected shape of data is %s, but %s for given data" % (shape_, X.shape[1:])
        logging.error(error_msg)
        raise ValueError(error_msg)


def _check_tensor_dim_shape(X, n_dims, shape_):
    """Validate both tensor dimensionality and sample shape.

    Parameters
    ----------
    X : ndarray of shape (n_samples, I_1, ..., I_N)
        Input tensor data.
    n_dims : int
        Expected number of dimensions.
    shape_ : tuple
        Expected per-sample shape.
    """
    _check_n_dim(X, n_dims)
    _check_shape(X, shape_)


class MPCA(BaseEstimator, TransformerMixin):
    """Multilinear Principal Component Analysis (MPCA) estimator.

    Parameters
    ----------
    explained_variance_ratio : float, default=0.97
        Target cumulative explained variance ratio per mode.
    max_iter : int, default=1
        Maximum number of alternating optimization iterations.
    vectorize : bool, default=False
        If ``True``, output projected tensors as vectors.
    n_components : int, optional
        Number of output features when ``vectorize=True``.
    output_shape : tuple of int, optional
        Number of components to keep per mode. If given, it overrides
        ``explained_variance_ratio`` and the output dimensions are set exactly.
        Default is None, i.e. the output shape is derived from
        ``explained_variance_ratio``.

    Attributes
    ----------
    proj_mats_ : list of ndarray
        Transposed projection matrices with shapes ``(P_i, I_i)``.
    idx_order_ : ndarray
        Feature ranking indices by descending projected variance.
    mean_ : ndarray
        Per-feature empirical mean of the training data.
    input_shape_ : tuple
        Input per-sample tensor shape.
    output_shape_ : tuple
        Output per-sample tensor shape after projection. Equals
        ``output_shape`` when given, otherwise determined by
        ``explained_variance_ratio``.
    explained_variance_ratio_ : tuple of float
        Achieved cumulative explained variance ratio per mode after fitting.

    References
    ----------
        Haiping Lu, K.N. Plataniotis, and A.N. Venetsanopoulos, "MPCA: Multilinear Principal Component Analysis of
        Tensor Objects", IEEE Transactions on Neural Networks, Vol. 19, No. 1, Page: 18-39, January 2008. For initial
        Matlab implementation, please go to https://uk.mathworks.com/matlabcentral/fileexchange/26168.

    Examples
    --------
        >>> import numpy as np
        >>> from kalelinear.transformer import MPCA
        >>> X = np.random.random((40, 20, 25, 20))
        >>> X.shape
        (40, 20, 25, 20)
        >>> mpca = MPCA()
        >>> X_projected = mpca.fit_transform(X)
        >>> X_projected.shape
        (40, 18, 23, 18)
        >>> X_projected = mpca.transform(X)
        >>> X_projected.shape
        (40, 7452)
        >>> X_projected = mpca.transform(X, vectorize=True)
        >>> X_projected.shape
        (40, 50)
        >>> X_reconstructed = mpca.inverse_transform(X_projected)
        >>> X_reconstructed.shape
        (40, 20, 25, 20)
    """

    def __init__(
        self, explained_variance_ratio=0.97, max_iter=1, vectorize=False, n_components=None, output_shape=None
    ):
        self.explained_variance_ratio = explained_variance_ratio
        if max_iter > 0 and isinstance(max_iter, int):
            self.max_iter = max_iter
        else:
            msg = "Number of max iterations must be a positive integer but given %s" % max_iter
            logging.error(msg)
            raise ValueError(msg)
        self.proj_mats_ = []
        self.vectorize = vectorize
        self.n_components = n_components
        if output_shape is None:
            self.output_shape = None
        elif isinstance(output_shape, (tuple, list)) and all(
            isinstance(v, (int, np.integer)) and v > 0 for v in output_shape
        ):
            self.output_shape = tuple(int(v) for v in output_shape)
        else:
            msg = "output_shape must be None or a sequence of positive integers but given %s" % (output_shape,)
            logging.error(msg)
            raise ValueError(msg)

    def fit(self, X, y=None):
        """Fit MPCA to tensor data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, I_1, ..., I_N)
            Input tensor samples.
        y : None, default=None
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        self : MPCA
            Fitted estimator.
        """
        self._fit(X)
        return self

    def _fit(self, X):
        """Internal solver for MPCA projection matrices.

        Parameters
        ----------
        X : ndarray of shape (n_samples, I_1, ..., I_N)
            Input tensor samples.

        Returns
        -------
        self : MPCA
            Fitted estimator.
        """

        shape_ = X.shape  # shape of input data
        n_samples = shape_[0]
        n_dims = X.ndim

        if n_samples < 2:
            error_msg = "MPCA requires at least 2 samples to fit."
            logging.error(error_msg)
            raise ValueError(error_msg)

        self.input_shape_ = shape_[1:]

        # Samples are processed in chunks so that a centered copy of the full
        # data never has to be materialized and disk-backed inputs (e.g. a
        # memory-mapped array or a loader over file paths) are read one chunk
        # at a time.
        n_features_in = int(np.prod(shape_[1:]))
        chunk_size = _chunk_size(n_samples, n_features_in)

        mean_acc = np.zeros(shape_[1:], dtype=np.float64)
        for start in range(0, n_samples, chunk_size):
            mean_acc += X[start : start + chunk_size].sum(axis=0, dtype=np.float64)
        self.mean_ = mean_acc / n_samples

        # init: accumulate per-mode covariance over chunked unfoldings
        covariance_matrices = {}
        for i in range(1, n_dims):
            mode_cov = np.zeros((shape_[i], shape_[i]))
            for start in range(0, n_samples, chunk_size):
                batch = X[start : start + chunk_size] - self.mean_
                batch_unfold = unfold(batch, mode=i)
                mode_cov += batch_unfold @ batch_unfold.T
            covariance_matrices[i] = mode_cov

        # get the output tensor shape: either user-specified or derived from the
        # cumulative distribution of eigen values for each mode
        if self.output_shape is not None:
            output_shape = self.output_shape
            if len(output_shape) != n_dims - 1:
                error_msg = "output_shape must have length %s (one entry per mode) but has %s" % (
                    n_dims - 1,
                    len(output_shape),
                )
                logging.error(error_msg)
                raise ValueError(error_msg)
            for mode_i, n_comp in enumerate(output_shape, start=1):
                if n_comp > shape_[mode_i]:
                    error_msg = "output_shape entry %s must not exceed the input size %s of mode %s but is %s" % (
                        mode_i - 1,
                        shape_[mode_i],
                        mode_i,
                        n_comp,
                    )
                    logging.error(error_msg)
                    raise ValueError(error_msg)
        else:
            output_shape = ()

        proj_matrices = []
        explained_variance_ratios = []
        for mode_i in range(1, n_dims):
            if self.output_shape is None:
                eigenvalues = eigvalsh(covariance_matrices[mode_i])
                idx_sorted = np.argsort(eigenvalues)[::-1]
                cum = eigenvalues[idx_sorted]
                tot_var = np.sum(cum)

                cum_var = np.cumsum(cum)
                mode_n_components = min(
                    int(np.searchsorted(cum_var, self.explained_variance_ratio * tot_var, side="right")) + 1,
                    shape_[mode_i],
                )
                output_shape += (mode_n_components,)
            else:
                mode_n_components = output_shape[mode_i - 1]

            # Only the top mode_n_components eigenvectors are needed; scipy eigh
            # returns them in ascending eigenvalue order for the requested range.
            subset_eigenvalues, eigenvectors = eigh(
                covariance_matrices[mode_i], subset_by_index=[shape_[mode_i] - mode_n_components, shape_[mode_i] - 1]
            )
            tot_var = np.trace(covariance_matrices[mode_i])
            explained_variance_ratio = subset_eigenvalues.sum() / tot_var if tot_var > 0 else 0.0
            explained_variance_ratios.append(explained_variance_ratio)
            proj_matrices.append(eigenvectors[:, ::-1].T)

        for _iter in range(self.max_iter):
            for mode_i in range(1, n_dims):  # ith mode
                mode_cov_mat = np.zeros((shape_[mode_i], shape_[mode_i]))
                proj_other = [proj_matrices[m] for m in range(n_dims - 1) if m != mode_i - 1]
                modes_other = [m for m in range(1, n_dims) if m != mode_i]
                for start in range(0, n_samples, chunk_size):
                    batch = X[start : start + chunk_size] - self.mean_
                    batch_proj = multi_mode_dot(batch, proj_other, modes=modes_other)
                    batch_unfold = unfold(batch_proj, mode=mode_i)
                    mode_cov_mat += batch_unfold @ batch_unfold.T

                _, eigenvectors = eigh(
                    mode_cov_mat,
                    subset_by_index=[shape_[mode_i] - output_shape[mode_i - 1], shape_[mode_i] - 1],
                )
                proj_matrices[mode_i - 1] = eigenvectors[:, ::-1].T

        # variance of the projected features, accumulated per chunk so the full
        # unfolded projection never has to be materialized
        x_proj_var = np.zeros(int(np.prod(output_shape)))
        modes_all = [m for m in range(1, n_dims)]
        for start in range(0, n_samples, chunk_size):
            batch = X[start : start + chunk_size] - self.mean_
            batch_proj = multi_mode_dot(batch, proj_matrices, modes=modes_all)
            batch_unfold = unfold(batch_proj, mode=0)  # unfold the chunked projection to shape (n_chunk, n_features)
            x_proj_var += np.einsum("ij,ij->j", batch_unfold, batch_unfold)
        idx_order_ = np.argsort(-x_proj_var)

        self.proj_mats_ = proj_matrices
        self.idx_order_ = idx_order_
        self.output_shape_ = output_shape
        self.explained_variance_ratio_ = tuple(explained_variance_ratios)
        self.n_dims_ = n_dims

        return self

    def transform(self, X, vectorize=None):
        """Project data to the MPCA subspace.

        Parameters
        ----------
        X : ndarray of shape (n_samples, I_1, ..., I_N) or (I_1, ..., I_N)
            Input tensor data.
        vectorize : bool, default=None
            Whether to return the projected data as vectors. If ``None``
            (default), the value set in ``__init__`` (``self.vectorize``) is
            used.

        Returns
        -------
        X_projected : ndarray
            Projected data. Shape is ``(n_samples, P_1, ..., P_N)`` when
            ``vectorize=False``. Otherwise returns vectorized features with
            optional truncation to ``n_components``.
        """
        if vectorize is None:
            vectorize = self.vectorize
        # reshape X to shape (1, I_1, I_2, ..., I_N) if X in shape (I_1, I_2, ..., I_N), i.e. n_samples = 1
        if X.ndim == self.n_dims_ - 1:
            X = X.reshape((1,) + X.shape)
        _check_tensor_dim_shape(X, self.n_dims_, self.input_shape_)
        X = X - self.mean_

        # projected tensor in lower dimensions
        X_projected = multi_mode_dot(X, self.proj_mats_, modes=[m for m in range(1, self.n_dims_)])

        n_components = self.n_components
        if vectorize:
            X_projected = unfold(X_projected, mode=0)
            X_projected = X_projected[:, self.idx_order_]
            if isinstance(n_components, int):
                n_features = int(np.prod(self.output_shape_))
                if n_components > n_features:
                    warn_msg = (
                        "n_components %d exceeds the maximum number, all features will be returned." % n_components
                    )
                    logging.warning(warn_msg)
                    warnings.warn(warn_msg)
                    n_components = n_features
                X_projected = X_projected[:, :n_components]

        return X_projected

    def inverse_transform(self, X):
        """Reconstruct original-space tensors from projected data.

        Parameters
        ----------
        X : ndarray
            Projected tensor data, either in tensor or vectorized format.

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, I_1, ..., I_N)
            Reconstructed tensor data in the original shape.
        """
        # reshape X to tensor in shape (n_samples, self.output_shape_) if X has been unfolded
        if X.ndim <= 2:
            if X.ndim == 1:
                # reshape X to a 2D matrix (1, n_components) if X in shape (n_components,)
                X = X.reshape((1, -1))
            n_samples = X.shape[0]
            n_features = X.shape[1]
            if n_features <= np.prod(self.output_shape_):
                x_ = np.zeros((n_samples, np.prod(self.output_shape_)))
                x_[:, self.idx_order_[:n_features]] = X[:]
            else:
                msg = "Feature dimension exceeds the shape upper limit."
                logging.error(msg)
                raise ValueError(msg)

            X = fold(x_, mode=0, shape=((n_samples,) + self.output_shape_))

        X_reconstructed = multi_mode_dot(X, self.proj_mats_, modes=[m for m in range(1, self.n_dims_)], transpose=True)

        X_reconstructed = X_reconstructed + self.mean_

        return X_reconstructed
