# =============================================================================
# @author: Shuo Zhou, The University of Sheffield
# @contact: shuo.zhou@sheffield.ac.uk
# =============================================================================
"""Common and Individual Feature Extraction (CIFE) implementation.

References
----------
Zhou, G., Cichocki, A., Zhang, Y. and Mandic, D., 2016. Group component
analysis for multiblock data: Common and individual feature extraction.
IEEE Transactions on Neural Networks and Learning Systems, 27(11),
pp.2426-2439.

The common orthogonal basis extraction (COBE) steps follow the authors'
reference implementations (pycifa and the accompanying MATLAB code).
"""

from numbers import Integral, Real

import numpy as np
from sklearn.utils._param_validation import Interval

from kalelinear.transformer._multiblock import _check_per_block_ranks, BaseCommonIndividualTransformer


def _column_space_basis(Y, pca_dim=None):
    """Return an orthonormal basis of the column space of ``Y`` (D x J)."""
    D, J = Y.shape
    U, s, _ = np.linalg.svd(Y, full_matrices=False)
    if s.size == 0 or s[0] == 0:
        raise ValueError("Each block must have a non-zero column space.")
    rank = int(np.sum(s > s[0] * max(D, J) * np.finfo(s.dtype).eps))
    if rank >= D:
        if pca_dim is None:
            raise ValueError(
                "A block spans the whole feature space, so common and individual "
                "subspaces cannot be separated. Reduce the dimensionality first "
                "or set `pca_dim` to truncate the per-block column spaces."
            )
        if 0 < pca_dim < 1:
            rank = min(int(np.floor(D * pca_dim)), D - 1)
        else:
            rank = min(int(pca_dim), D - 1)
        rank = max(rank, 1)
    return U[:, :rank], rank


def _cobe_common_basis(blocks, c, max_iter, tol, epsilon, pca_dim, random_state):
    """Extract a common orthogonal basis shared by all blocks.

    Parameters
    ----------
    blocks : list of ndarray of shape (D, J_n)
        Data blocks sharing the same feature dimension ``D``.
    c : int or None
        Number of common components. When None, the number is estimated from
        the residual ``epsilon`` criterion.
    max_iter : int
        Maximum power iterations per common direction.
    tol : float
        Convergence tolerance for the power iterations.
    epsilon : float
        Residual threshold below which a direction counts as common.
    pca_dim : int, float or None
        Optional per-block dimensionality truncation for blocks spanning the
        whole feature space.
    random_state : RandomState
        Random number generator for initializing the power iterations.

    Returns
    -------
    common_basis : ndarray of shape (D, n_common)
        Orthonormal common basis shared by all blocks.
    """
    n_blocks = len(blocks)
    D = blocks[0].shape[0]
    bases = []
    ranks = []
    for Y in blocks:
        basis, rank = _column_space_basis(Y, pca_dim=pca_dim)
        bases.append(basis)
        ranks.append(rank)
    min_rank = min(ranks)
    if min_rank == 0:
        return np.zeros((D, 0))
    if c is not None and c <= 0:
        return np.zeros((D, 0))

    order = np.argsort(ranks)
    projections = [np.zeros((bases[n].shape[1], min_rank)) for n in range(n_blocks)]
    common_basis = np.zeros((D, min_rank))
    residuals = []

    def _power_iteration(initial, column):
        direction = initial / np.linalg.norm(initial)
        for _ in range(max_iter):
            previous = direction
            update = np.zeros(D)
            for n in range(n_blocks):
                projections[n][:, column] = bases[n].T @ direction
                update += bases[n] @ projections[n][:, column]
            update_norm = np.linalg.norm(update)
            if update_norm == 0:
                break
            direction = update / update_norm
            if abs(previous @ direction) > 1 - tol:
                break
        return direction

    # Seek the first common direction.
    initial = bases[order[0]] @ random_state.randn(bases[order[0]].shape[1])
    first = _power_iteration(initial, 0)
    residual = 0.0
    for n in range(n_blocks):
        projection = bases[n].T @ first
        residual += 1 - projection @ projection
    residual /= n_blocks
    residuals.append(residual)

    if c is None and residual > epsilon:
        return np.zeros((D, 0))

    if c is not None:
        c = min(c, min_rank)
        common_basis = np.zeros((D, c))
        common_basis[:, 0] = first
        residuals.extend([np.inf] * (c - 1))
    else:
        common_basis[:, 0] = first
        residuals.extend([np.inf] * (min_rank - 1))

    # Seek the remaining common directions with deflation.
    for j in range(1, min_rank):
        if c is not None and j >= c:
            break
        for n in range(n_blocks):
            basis = bases[n]
            bases[n] = basis - np.outer(basis @ projections[n][:, j - 1], projections[n][:, j - 1])
        initial = bases[order[0]] @ random_state.randn(bases[order[0]].shape[1])
        direction = _power_iteration(initial, j)
        residual = 0.0
        for n in range(n_blocks):
            projection = bases[n].T @ direction
            residual += 1 - projection @ projection
        residual /= n_blocks
        residuals[j] = residual
        if c is None and residual > epsilon:
            residuals[j] = np.inf
            break
        common_basis[:, j] = direction

    common_basis = common_basis[:, ~np.isinf(np.asarray(residuals))]
    if common_basis.shape[1] > 0:
        u, _, vt = np.linalg.svd(common_basis, full_matrices=False)
        common_basis = u @ vt
    return common_basis


class CIFE(BaseCommonIndividualTransformer):
    """Common and Individual Feature Extraction (CIFE).

    CIFE decomposes multiblock data into a common feature subspace shared by
    all blocks and block-specific individual subspaces. The common subspace is
    extracted with the common orthogonal basis extraction (COBE) algorithm and
    the individual subspaces are obtained from the residual of each block after
    removing its common part.

    ``X`` can be a single matrix with a ``groups`` array of block ids, or a
    list of block matrices that all share the same feature space.
    :meth:`transform` returns the common feature scores of the samples while
    :meth:`transform_individual` returns the block-specific individual scores.

    Parameters
    ----------
    n_common_components : int or None, default=None
        Number of common components to extract. When None, the number is
        estimated automatically from the residual threshold ``epsilon``.
    n_individual_components : int, array-like or None, default=None
        Number of individual components per block. When None, all numerically
        non-zero residual directions are kept.
    max_iter : int, default=200
        Maximum power iterations for each common direction.
    tol : float, default=1e-6
        Convergence tolerance for the power iterations.
    epsilon : float, default=0.01
        Residual threshold used to decide whether a direction is common when
        ``n_common_components`` is None.
    pca_dim : int, float or None, default=None
        Optional truncation of per-block column spaces, either as a relative
        fraction in (0, 1) or an absolute number of components. Required when
        a block spans the whole feature space.
    random_state : int, RandomState or None, default=None
        Random seed for initializing the power iterations.

    Attributes
    ----------
    common_components_ : ndarray of shape (n_features, n_common_components_)
        Orthonormal common feature-space basis.
    individual_components_ : list of ndarray of shape (n_features, rank)
        Per-block orthonormal individual feature-space bases.
    common_scores_ : list of ndarray of shape (n_samples_in_block, n_common_components_)
        Common scores of the training samples in each block.
    individual_scores_ : list of ndarray of shape (n_samples_in_block, rank)
        Individual scores of the training samples in each block.
    individual_ranks_ : ndarray of shape (n_blocks,)
        Number of individual components retained per block.
    n_common_components_ : int
        Number of common components retained.
    """

    _parameter_constraints: dict = {
        **BaseCommonIndividualTransformer._parameter_constraints,
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "epsilon": [Interval(Real, 0, None, closed="left")],
        "pca_dim": [
            Interval(Real, 0, 1, closed="right"),
            Interval(Integral, 2, None, closed="left"),
            None,
        ],
    }

    def __init__(
        self,
        n_common_components=None,
        n_individual_components=None,
        max_iter=200,
        tol=1e-6,
        epsilon=0.01,
        pca_dim=None,
        random_state=None,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = epsilon
        self.pca_dim = pca_dim
        super().__init__(
            n_common_components=n_common_components,
            n_individual_components=n_individual_components,
            random_state=random_state,
        )

    def _fit_blocks(self, blocks):
        D = self.n_features_in_
        transposed = [block.T for block in blocks]
        common_components = _cobe_common_basis(
            transposed,
            c=self.n_common_components,
            max_iter=self.max_iter,
            tol=self.tol,
            epsilon=self.epsilon,
            pca_dim=self.pca_dim,
            random_state=self.random_state_,
        )
        n_common = common_components.shape[1]
        common_scores = [block @ common_components for block in blocks]

        ranks_spec = _check_per_block_ranks(self.n_individual_components, self.n_blocks_, "n_individual_components")
        individual_components = []
        individual_scores = []
        individual_ranks = []
        for n, (X_n, Y_n) in enumerate(zip(blocks, transposed)):
            residual = Y_n - common_components @ (common_components.T @ Y_n)
            U_i, s_i, _ = np.linalg.svd(residual, full_matrices=False)
            if ranks_spec is None:
                if s_i.size == 0:
                    rank = 0
                else:
                    noise_tol = s_i[0] * max(residual.shape) * np.finfo(s_i.dtype).eps
                    rank = int(np.sum(s_i > noise_tol))
            else:
                rank = min(int(ranks_spec[n]), U_i.shape[1])
            if rank > 0:
                U_i = U_i[:, :rank]
                individual_components.append(U_i)
                individual_scores.append(X_n @ U_i)
            else:
                individual_components.append(np.zeros((D, 0)))
                individual_scores.append(np.zeros((X_n.shape[0], 0)))
            individual_ranks.append(rank)

        self.common_components_ = common_components
        self.common_scores_ = common_scores
        self.individual_components_ = individual_components
        self.individual_scores_ = individual_scores
        self.individual_ranks_ = np.asarray(individual_ranks, dtype=int)
        self.n_common_components_ = n_common
