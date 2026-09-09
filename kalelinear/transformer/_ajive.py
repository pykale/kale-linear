# =============================================================================
# @author: Shuo Zhou, The University of Sheffield
# @contact: shuo.zhou@sheffield.ac.uk
# =============================================================================
"""Angle-based Joint and Individual Variation Explained (AJIVE) implementation.

References
----------
Feng, Q., Jiang, M., Hannig, J. and Marron, J.S., 2018. Angle-based joint and
individual variation explained. Journal of Multivariate Analysis, 166,
pp.241-265.

The algorithm follows the authors' reference MATLAB implementation
(MeileiJiang/AJIVE_Project on GitHub) with the same perturbation-bound and rank
selection steps.
"""

from numbers import Integral, Real

import numpy as np
from sklearn.utils._param_validation import Interval

from kalelinear.transformer._multiblock import _check_per_block_ranks, BaseCommonIndividualTransformer

_FERROR = 1e-10


def _jive_rand_null_norm(data, basis, n_sim, random_state):
    """Operator norm of data on random directions orthogonal to ``basis``."""
    n_ambient_dims = basis.shape[0]
    n_null_dims = n_ambient_dims - basis.shape[1]
    if n_null_dims <= 0:
        return np.zeros(n_sim)
    # ``initial_ranks`` may legally exceed the null-space dimension, in which
    # case requesting one orthogonal direction per basis vector would run out
    # of genuine null directions: once ``current`` spans the ambient space the
    # retry normalises round-off noise (or never terminates in exact
    # arithmetic). Only ask for as many directions as the null space can hold.
    n_directions = min(basis.shape[1], n_null_dims)
    null_norms = np.empty(n_sim)
    for i in range(n_sim):
        current = basis.copy()
        directions = []
        for _ in range(n_directions):
            direction = random_state.randn(n_ambient_dims)
            direction = direction - current @ (current.T @ direction)
            norm = np.linalg.norm(direction)
            while norm == 0:
                direction = random_state.randn(n_ambient_dims)
                direction = direction - current @ (current.T @ direction)
                norm = np.linalg.norm(direction)
            direction /= norm
            directions.append(direction)
            current = np.column_stack((current, direction))
        directions = np.column_stack(directions)
        # The Wedin bound needs the spectral (2-)norm, matching the reference
        # implementation's MATLAB ``norm(data * nulldir)``; the default
        # Frobenius norm grows with the number of sampled directions and would
        # systematically inflate the angle bound.
        null_norms[i] = np.linalg.norm(data @ directions, ord=2)
    return null_norms


def _wedin_angle_bound(block, n_sim, U, S, V, random_state):
    """Resampled Wedin perturbation-angle bound for one data block."""
    delta = S[-1]
    if delta <= S[0] * np.finfo(float).eps:
        raise ValueError(
            "`initial_ranks` exceeds the numerical rank of a data block: the "
            "smallest retained singular value is zero up to machine precision. "
            "Lower `initial_ranks` for that block."
        )
    row_bound = _jive_rand_null_norm(block, V, n_sim, random_state)
    column_bound = _jive_rand_null_norm(block.T, U, n_sim, random_state)
    ratio = np.maximum(row_bound, column_bound) / delta
    ratio = np.clip(ratio, 0.0, 1.0)
    return np.rad2deg(np.arcsin(ratio))


def _random_direction_ssv(n, ranks, n_sim, random_state):
    """Largest squared singular values of random stacked subspaces."""
    stacked = np.zeros((int(np.sum(ranks)), n))
    values = np.empty(n_sim)
    for i in range(n_sim):
        row = 0
        for rank in ranks:
            Q, _ = np.linalg.qr(random_state.randn(n, rank))
            stacked[row : row + rank] = Q.T
            row += rank
        values[i] = np.linalg.norm(stacked, 2) ** 2
    return values


def _select_joint_threshold(wedin_ssv_bounds, percentile, random_ssv_bound):
    """Joint-rank threshold following the reference AJIVEJointSelectMJ logic.

    The random-direction bound is compared with the fixed 5th percentile of
    the Wedin bounds and, when it is larger, used alone. Otherwise the Wedin
    bound at the requested ``percentile`` applies.
    """
    if random_ssv_bound > np.percentile(wedin_ssv_bounds, 5):
        return random_ssv_bound
    return np.percentile(wedin_ssv_bounds, percentile)


class AJIVE(BaseCommonIndividualTransformer):
    """Angle-based Joint and Individual Variation Explained (AJIVE).

    AJIVE decomposes multiblock data into a joint (common) subspace shared by
    all blocks and block-specific individual subspaces. It estimates the joint
    rank from the Wedin perturbation bound of the stacked per-block row spaces
    and reconstructs the joint and individual structures with low-rank SVDs.

    ``X`` can be a single matrix with a ``groups`` array of block ids, or a
    list of block matrices that all share the same feature space.
    :meth:`transform` returns the common feature scores of the samples while
    :meth:`transform_individual` returns the block-specific individual scores.

    Parameters
    ----------
    n_common_components : int or None, default=None
        Number of joint components. When None, the joint rank is selected from
        the Wedin perturbation bound.
    n_individual_components : int, array-like or None, default=None
        Number of individual components per block. When None, components above
        the per-block singular-value threshold are kept.
    initial_ranks : int, array-like or None, default=None
        Initial signal rank of each block. When None, ranks are estimated from
        the fraction of variance explained by ``variance_threshold``.
    variance_threshold : float, default=0.95
        Cumulative explained-variance fraction used to estimate ``initial_ranks``
        when it is None.
    n_resamples : int, default=1000
        Number of re-samples for the Wedin perturbation bound.
    percentile : float, default=5
        Percentile of the Wedin bounds used for the joint rank selection.
        Applied only when the Wedin bound dominates the random-direction
        bound; otherwise the random-direction bound alone sets the threshold
        (as in the reference implementation).
    random_state : int, RandomState or None, default=None
        Random seed for the perturbation-bound re-sampling.

    Attributes
    ----------
    common_components_ : ndarray of shape (n_features, n_common_components_)
        Orthonormal joint feature-space basis.
    individual_components_ : list of ndarray of shape (n_features, rank)
        Per-block orthonormal individual feature-space bases.
    common_scores_ : list of ndarray of shape (n_samples_in_block, n_common_components_)
        Common scores of the training samples in each block.
    individual_scores_ : list of ndarray of shape (n_samples_in_block, rank)
        Individual scores of the training samples in each block.
    individual_ranks_ : ndarray of shape (n_blocks,)
        Number of individual components retained per block.
    n_common_components_ : int
        Number of joint components retained.
    """

    _parameter_constraints: dict = {
        **BaseCommonIndividualTransformer._parameter_constraints,
        "initial_ranks": ["array-like", Interval(Integral, 1, None, closed="left"), None],
        "variance_threshold": [Interval(Real, 0, 1, closed="both")],
        "n_resamples": [Interval(Integral, 1, None, closed="left")],
        "percentile": [Interval(Real, 0, 100, closed="both")],
    }

    def __init__(
        self,
        n_common_components=None,
        n_individual_components=None,
        initial_ranks=None,
        variance_threshold=0.95,
        n_resamples=1000,
        percentile=5,
        random_state=None,
    ):
        self.initial_ranks = initial_ranks
        self.variance_threshold = variance_threshold
        self.n_resamples = n_resamples
        self.percentile = percentile
        super().__init__(
            n_common_components=n_common_components,
            n_individual_components=n_individual_components,
            random_state=random_state,
        )

    def _resolve_initial_ranks(self, blocks):
        if self.initial_ranks is None:
            ranks = []
            for block in blocks:
                singular_values = np.linalg.svd(block, compute_uv=False)
                if singular_values.size == 0 or singular_values[0] == 0:
                    # A zero-energy block has no signal rank; the positive-rank
                    # validation below then rejects it with a clear message.
                    ranks.append(0)
                    continue
                explained = np.cumsum(singular_values**2) / np.sum(singular_values**2)
                rank = int(np.searchsorted(explained, self.variance_threshold) + 1)
                # Near variance_threshold == 1 the cumulative sum can round to
                # just below one, making ``searchsorted`` return the full size
                # and include machine-noise singular values. Cap the rank by the
                # numerical rank so the estimate is stable across BLAS/NumPy
                # round-off and never trips the Wedin bound's rank guard.
                numerical_rank = int(
                    np.sum(singular_values > singular_values[0] * max(block.shape) * np.finfo(float).eps)
                )
                ranks.append(int(min(rank, numerical_rank)))
            return np.asarray(ranks, dtype=int)
        ranks = _check_per_block_ranks(self.initial_ranks, self.n_blocks_, "initial_ranks")
        for n, (rank, block) in enumerate(zip(ranks, blocks)):
            if rank < 1 or rank > min(block.shape):
                raise ValueError(
                    f"`initial_ranks[{n}]` must be between 1 and min(samples, features) "
                    f"= {min(block.shape)}, got {rank}."
                )
            if not np.any(block):
                # A zero-energy block has no row or column space to share: the
                # perturbation bound would divide by a zero singular value and
                # every residual direction would be misclassified as individual
                # signal. Reject it even when the ranks were supplied explicitly.
                raise ValueError(f"Block {n} has zero energy, so `initial_ranks[{n}]` must not be positive.")
        return ranks

    def _fit_blocks(self, blocks):
        D = self.n_features_in_
        ranks = self._resolve_initial_ranks(blocks)
        if np.any(ranks < 1):
            raise ValueError("`initial_ranks` must contain positive values for every block.")

        stacked = np.zeros((int(np.sum(ranks)), D))
        thresholds = np.empty(self.n_blocks_)
        angle_bounds = []
        row = 0
        for n, block in enumerate(blocks):
            rank = ranks[n]
            U, s, Vt = np.linalg.svd(block, full_matrices=False)
            if rank + 1 <= len(s):
                thresholds[n] = 0.5 * (s[rank - 1] + s[rank])
            else:
                thresholds[n] = 0.5 * s[-1]
            U0, S0, V0 = U[:, :rank], s[:rank], Vt[:rank].T
            stacked[row : row + rank] = V0.T
            row += rank
            angle_bounds.append(_wedin_angle_bound(block, self.n_resamples, U0, S0, V0, self.random_state_))
        angle_bounds = np.vstack(angle_bounds)

        _, s_stacked, Vt_stacked = np.linalg.svd(stacked, full_matrices=False)
        max_joint_rank = int(np.min(ranks))
        # The joint space is a subspace of every block's row space, so it can
        # have at most ``max_joint_rank`` dimensions. Keep only the leading
        # singular values and vectors, as the reference implementation computes
        # only ``min(vecr)`` singular vectors; directions supported by only a
        # subset of blocks are then never candidates for the common subspace.
        s_stacked = s_stacked[:max_joint_rank]
        Vt_stacked = Vt_stacked[:max_joint_rank]
        wedin_ssv_bounds = np.maximum(np.sum(np.cos(np.deg2rad(angle_bounds)) ** 2, axis=0), 1.0)
        random_ssvs = _random_direction_ssv(D, ranks, 100, self.random_state_)
        random_ssv_bound = np.percentile(random_ssvs, 95)
        joint_threshold = _select_joint_threshold(wedin_ssv_bounds, self.percentile, random_ssv_bound)
        joint_rank = int(np.sum(s_stacked**2 + _FERROR > joint_threshold))
        if self.n_common_components is not None:
            joint_rank = min(int(self.n_common_components), len(s_stacked))

        row_joint = Vt_stacked[:joint_rank]
        drop_rows = set()
        for n, block in enumerate(blocks):
            projected = block @ row_joint.T
            # The slack must be relative to the block threshold: an absolute
            # one would dominate once the data is rescaled below it and drop
            # every candidate joint row.
            low_variance = np.flatnonzero(np.sqrt(np.sum(projected**2, axis=0)) <= thresholds[n] * (1 + _FERROR))
            drop_rows.update(low_variance.tolist())
        if drop_rows:
            keep_rows = [j for j in range(row_joint.shape[0]) if j not in drop_rows]
            row_joint = row_joint[keep_rows]
        joint_rank = row_joint.shape[0]

        common_components = row_joint.T
        common_scores = [block @ common_components for block in blocks]

        ranks_spec = _check_per_block_ranks(self.n_individual_components, self.n_blocks_, "n_individual_components")
        individual_components = []
        individual_scores = []
        individual_ranks = []
        for n, block in enumerate(blocks):
            individual = block - block @ common_components @ common_components.T
            s_individual = np.linalg.svd(individual, compute_uv=False)
            if ranks_spec is None:
                # Relative slack keeps the individual rank estimate invariant
                # to the data scale.
                rank = int(np.sum(s_individual * (1 + _FERROR) > thresholds[n]))
            else:
                rank = min(int(ranks_spec[n]), len(s_individual))
            rank = max(rank, 0)
            if rank > 0:
                U_i, S_i, Vt_i = np.linalg.svd(individual, full_matrices=False)
                U_i, S_i, Vt_i = U_i[:, :rank], S_i[:rank], Vt_i[:rank]
                individual_components.append(Vt_i.T)
                individual_scores.append(U_i * S_i)
            else:
                individual_components.append(np.zeros((D, 0)))
                individual_scores.append(np.zeros((block.shape[0], 0)))
            individual_ranks.append(rank)

        self.common_components_ = common_components
        self.common_scores_ = common_scores
        self.individual_components_ = individual_components
        self.individual_scores_ = individual_scores
        self.individual_ranks_ = np.asarray(individual_ranks, dtype=int)
        self.n_common_components_ = joint_rank
