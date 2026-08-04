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
    var_ratio : float, default=0.97
        Target cumulative explained variance ratio per mode.
    max_iter : int, default=1
        Maximum number of alternating optimization iterations.
    vectorize : bool, default=False
        If ``True``, output projected tensors as vectors.
    n_components : int, optional
        Number of output features when ``vectorize=True``.

    Attributes
    ----------
    proj_mats : list of ndarray
        Transposed projection matrices with shapes ``(P_i, I_i)``.
    idx_order : ndarray
        Feature ranking indices by descending projected variance.
    mean_ : ndarray
        Per-feature empirical mean of the training data.
    sample_shape : tuple
        Input per-sample tensor shape.
    modewise_n_components : tuple
        Output per-sample tensor shape after projection.

    References
    ----------
        Haiping Lu, K.N. Plataniotis, and A.N. Venetsanopoulos, "MPCA: Multilinear Principal Component Analysis of
        Tensor Objects", IEEE Transactions on Neural Networks, Vol. 19, No. 1, Page: 18-39, January 2008. For initial
        Matlab implementation, please go to https://uk.mathworks.com/matlabcentral/fileexchange/26168.

    Examples
    --------
        >>> import numpy as np
        >>> from kalelinear.transformer import MPCA
        >>> x = np.random.random((40, 20, 25, 20))
        >>> x.shape
        (40, 20, 25, 20)
        >>> mpca = MPCA()
        >>> x_projected = mpca.fit_transform(x)
        >>> x_projected.shape
        (40, 18, 23, 18)
        >>> x_projected = mpca.transform(x)
        >>> x_projected.shape
        (40, 7452)
        >>> x_projected = mpca.transform(x)
        >>> x_projected.shape
        (40, 50)
        >>> x_rec = mpca.inverse_transform(x_projected)
        >>> x_rec.shape
        (40, 20, 25, 20)
    """

    def __init__(self, var_ratio=0.97, max_iter=1, vectorize=False, n_components=None):
        self.var_ratio = var_ratio
        if max_iter > 0 and isinstance(max_iter, int):
            self.max_iter = max_iter
        else:
            msg = "Number of max iterations must be a positive integer but given %s" % max_iter
            logging.error(msg)
            raise ValueError(msg)
        self.proj_mats = []
        self.vectorize = vectorize
        self.n_components = n_components

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

        self.sample_shape = shape_[1:]

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

        # get the output tensor shape based on the cumulative distribution of eigen values for each mode
        modewise_n_components = ()
        proj_matrices = []
        for mode_i in range(1, n_dims):
            eigenvalues = eigvalsh(covariance_matrices[mode_i])
            idx_sorted = np.argsort(eigenvalues)[::-1]
            cum = eigenvalues[idx_sorted]
            tot_var = np.sum(cum)

            cum_var = np.cumsum(cum)
            mode_n_components = min(
                int(np.searchsorted(cum_var, self.var_ratio * tot_var, side="right")) + 1, shape_[mode_i]
            )
            modewise_n_components += (mode_n_components,)

            # Only the j largest eigenvectors are needed; scipy eigh returns them in
            # ascending eigenvalue order for the requested index range.
            _, eigenvectors = eigh(
                covariance_matrices[mode_i], subset_by_index=[shape_[mode_i] - mode_n_components, shape_[mode_i] - 1]
            )
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
                    subset_by_index=[shape_[mode_i] - modewise_n_components[mode_i - 1], shape_[mode_i] - 1],
                )
                proj_matrices[mode_i - 1] = eigenvectors[:, ::-1].T

        # variance of the projected features, accumulated per chunk so the full
        # unfolded projection never has to be materialized
        x_proj_var = np.zeros(int(np.prod(modewise_n_components)))
        modes_all = [m for m in range(1, n_dims)]
        for start in range(0, n_samples, chunk_size):
            batch = X[start : start + chunk_size] - self.mean_
            batch_proj = multi_mode_dot(batch, proj_matrices, modes=modes_all)
            batch_unfold = unfold(batch_proj, mode=0)  # unfold the chunked projection to shape (n_chunk, n_features)
            x_proj_var += np.einsum("ij,ij->j", batch_unfold, batch_unfold)
        idx_order = np.argsort(-x_proj_var)

        self.proj_mats = proj_matrices
        self.idx_order = idx_order
        self.modewise_n_components = modewise_n_components
        self.n_dims = n_dims

        return self

    def transform(self, X):
        """Project data to the MPCA subspace.

        Parameters
        ----------
        X : ndarray of shape (n_samples, I_1, ..., I_N) or (I_1, ..., I_N)
            Input tensor data.

        Returns
        -------
        x_projected : ndarray
            Projected data. Shape is ``(n_samples, P_1, ..., P_N)`` when
            ``vectorize=False``. Otherwise returns vectorized features with
            optional truncation to ``n_components``.
        """
        # reshape X to shape (1, I_1, I_2, ..., I_N) if X in shape (I_1, I_2, ..., I_N), i.e. n_samples = 1
        if X.ndim == self.n_dims - 1:
            X = X.reshape((1,) + X.shape)
        _check_tensor_dim_shape(X, self.n_dims, self.sample_shape)
        X = X - self.mean_

        # projected tensor in lower dimensions
        x_projected = multi_mode_dot(X, self.proj_mats, modes=[m for m in range(1, self.n_dims)])

        n_components = self.n_components
        if self.vectorize:
            x_projected = unfold(x_projected, mode=0)
            x_projected = x_projected[:, self.idx_order]
            if isinstance(n_components, int):
                n_features = int(np.prod(self.modewise_n_components))
                if n_components > n_features:
                    warn_msg = (
                        "n_components %d exceeds the maximum number, all features will be returned." % n_components
                    )
                    logging.warning(warn_msg)
                    warnings.warn(warn_msg)
                    n_components = n_features
                x_projected = x_projected[:, :n_components]

        return x_projected

    def inverse_transform(self, X):
        """Reconstruct original-space tensors from projected data.

        Parameters
        ----------
        X : ndarray
            Projected tensor data, either in tensor or vectorized format.

        Returns
        -------
        x_rec : ndarray of shape (n_samples, I_1, ..., I_N)
            Reconstructed tensor data in the original shape.
        """
        # reshape X to tensor in shape (n_samples, self.modewise_n_components) if X has been unfolded
        if X.ndim <= 2:
            if X.ndim == 1:
                # reshape X to a 2D matrix (1, n_components) if X in shape (n_components,)
                X = X.reshape((1, -1))
            n_samples = X.shape[0]
            n_features = X.shape[1]
            if n_features <= np.prod(self.modewise_n_components):
                x_ = np.zeros((n_samples, np.prod(self.modewise_n_components)))
                x_[:, self.idx_order[:n_features]] = X[:]
            else:
                msg = "Feature dimension exceeds the shape upper limit."
                logging.error(msg)
                raise ValueError(msg)

            X = fold(x_, mode=0, shape=((n_samples,) + self.modewise_n_components))

        x_rec = multi_mode_dot(X, self.proj_mats, modes=[m for m in range(1, self.n_dims)], transpose=True)

        x_rec = x_rec + self.mean_

        return x_rec
