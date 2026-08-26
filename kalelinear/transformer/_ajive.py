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
    """Energy of data on random directions orthogonal to ``basis``."""
    n_ambient_dims = basis.shape[0]
    if basis.shape[1] >= n_ambient_dims:
        return np.zeros(n_sim)
    null_norms = np.empty(n_sim)
    for i in range(n_sim):
        current = basis.copy()
        directions = []
        for _ in range(basis.shape[1]):
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
        null_norms[i] = np.linalg.norm(data @ directions)
    return null_norms


def _wedin_angle_bound(block, n_sim, U, S, V, random_state):
    """Resampled Wedin perturbation-angle bound for one data block."""
    row_bound = _jive_rand_null_norm(block, V, n_sim, random_state)
    column_bound = _jive_rand_null_norm(block.T, U, n_sim, random_state)
    delta = S[-1]
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
                if singular_values.size == 0:
                    ranks.append(0)
                    continue
                explained = np.cumsum(singular_values**2) / np.sum(singular_values**2)
                rank = int(np.searchsorted(explained, self.variance_threshold) + 1)
                ranks.append(int(min(rank, singular_values.size)))
            return np.asarray(ranks, dtype=int)
        ranks = _check_per_block_ranks(self.initial_ranks, self.n_blocks_, "initial_ranks")
        for n, (rank, block) in enumerate(zip(ranks, blocks)):
            if rank < 1 or rank > min(block.shape):
                raise ValueError(
                    f"`initial_ranks[{n}]` must be between 1 and min(samples, features) "
                    f"= {min(block.shape)}, got {rank}."
                )
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
        s_stacked = s_stacked[: min(stacked.shape)]
        wedin_ssv_bounds = np.maximum(np.sum(np.cos(np.deg2rad(angle_bounds)) ** 2, axis=0), 1.0)
        wedin_ssv_bound = np.percentile(wedin_ssv_bounds, self.percentile)
        random_ssvs = _random_direction_ssv(D, ranks, 100, self.random_state_)
        random_ssv_bound = np.percentile(random_ssvs, 95)
        # Take the more conservative (larger) of the two perturbation bounds,
        # following the reference implementation: max(wedin, random).
        joint_threshold = max(wedin_ssv_bound, random_ssv_bound)
        joint_rank = int(np.sum(s_stacked**2 + _FERROR > joint_threshold))
        if self.n_common_components is not None:
            joint_rank = min(int(self.n_common_components), len(s_stacked))

        row_joint = Vt_stacked[:joint_rank]
        drop_rows = set()
        for n, block in enumerate(blocks):
            projected = block @ row_joint.T
            low_variance = np.flatnonzero(np.sqrt(np.sum(projected**2, axis=0)) <= thresholds[n] + _FERROR)
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
                rank = int(np.sum(s_individual + _FERROR > thresholds[n]))
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
