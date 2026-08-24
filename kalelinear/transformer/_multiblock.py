# =============================================================================
# @author: Shuo Zhou, Lalu Muhammad Riza Rizky, The University of Sheffield
# @contact: shuo.zhou@sheffield.ac.uk
# =============================================================================
"""Shared base classes for multiblock common and individual feature transformers."""

from abc import abstractmethod
from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, check_random_state


def _check_multiblock_input(X, groups=None):
    """Validate a multiblock input and return one matrix per block.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or list of array-like
        When ``X`` is a single matrix, ``groups`` must give the block id of
        each row. When ``X`` is a list, each element is one block and all
        blocks must share the same feature space (columns).
    groups : array-like of shape (n_samples,), default=None
        Block id for each sample when ``X`` is a single stacked matrix.

    Returns
    -------
    blocks : list of ndarray of shape (n_samples_in_block, n_features)
    groups : ndarray of shape (n_samples,) or None
    """
    if isinstance(X, (list, tuple)):
        if groups is not None:
            raise ValueError("`groups` must be None when `X` is a list of blocks.")
        if len(X) == 0:
            raise ValueError("`X` must contain at least one block.")
        blocks = []
        n_features = None
        for block in X:
            block = np.asarray(block, dtype=float)
            if block.ndim != 2:
                raise ValueError("Each block in `X` must be a 2D array.")
            if block.shape[0] == 0:
                raise ValueError("Each block in `X` must contain at least one sample.")
            if n_features is None:
                n_features = block.shape[1]
            elif block.shape[1] != n_features:
                raise ValueError("All blocks must share the same number of features.")
            blocks.append(block)
        if len(blocks) < 2:
            raise ValueError("At least two blocks are required for common and individual feature extraction.")
        return blocks, None

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("`X` must be a 2D array or a list of 2D block arrays.")
    if groups is None:
        raise ValueError("`groups` must be provided when `X` is a single stacked matrix.")
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.shape[0] != X.shape[0]:
        raise ValueError("`groups` must be a 1D array aligned with the rows of `X`.")
    block_ids = np.unique(groups)
    blocks = [X[groups == block_id] for block_id in block_ids]
    if len(blocks) < 2:
        raise ValueError("At least two blocks are required for common and individual feature extraction.")
    if any(block.shape[0] == 0 for block in blocks):
        raise ValueError("Each block must contain at least one sample.")
    return blocks, groups


def _check_per_block_ranks(n_components, n_blocks, name):
    """Validate an integer or a per-block sequence of component counts."""
    if n_components is None:
        return None
    if isinstance(n_components, (Integral, np.integer)):
        return np.full(n_blocks, int(n_components), dtype=int)
    ranks = np.asarray(n_components)
    if ranks.ndim != 1 or ranks.shape[0] != n_blocks:
        raise ValueError(f"{name} must be an integer or a sequence with one value per block.")
    if not np.issubdtype(ranks.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values.")
    if np.any(np.isnan(ranks)):
        raise ValueError(f"{name} must not contain NaN values.")
    if np.any(ranks < 0):
        raise ValueError(f"{name} must contain non-negative values.")
    if not np.all(np.equal(ranks, np.floor(ranks))):
        raise ValueError(f"{name} must contain integer values.")
    return ranks.astype(int)


class BaseCommonIndividualTransformer(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Base class for multiblock common and individual feature transformers.

    Subclasses learn a common feature subspace shared by all data blocks plus
    block-specific individual subspaces, following the common and individual
    feature extraction (CIFE) framework of Zhou and Cichocki (2016).

    The input convention follows the rest of ``kalelinear``: ``X`` is a sample
    matrix whose rows are partitioned into blocks by ``groups``, or a list of
    block matrices that all share the same feature space.
    """

    _parameter_constraints: dict = {
        "n_common_components": [Interval(Integral, 0, None, closed="left"), None],
        "n_individual_components": ["array-like", Interval(Integral, 0, None, closed="left"), None],
        "random_state": ["random_state"],
    }

    def __init__(self, n_common_components=None, n_individual_components=None, random_state=None):
        self.n_common_components = n_common_components
        self.n_individual_components = n_individual_components
        self.random_state = random_state

    def fit(self, X, y=None, groups=None, **fit_params):
        """Fit the transformer on multiblock data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of array-like
            Stacked samples from all blocks, or a list of block matrices
            sharing the same feature space.
        y : array-like of shape (n_samples,), default=None
            Ignored. Present for scikit-learn API consistency.
        groups : array-like of shape (n_samples,), default=None
            Block id for each sample, required when ``X`` is a single stacked
            matrix.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        self._validate_params()
        blocks, groups = _check_multiblock_input(X, groups)
        self.n_features_in_ = blocks[0].shape[1]
        self.n_blocks_ = len(blocks)
        self.block_sizes_ = np.array([block.shape[0] for block in blocks])
        self.random_state_ = check_random_state(self.random_state)
        self._fit_blocks(blocks)
        self._n_features_out = self.n_common_components_
        return self

    @abstractmethod
    def _fit_blocks(self, blocks):
        """Run the algorithm on validated per-block matrices."""

    def transform(self, X, groups=None):
        """Project samples onto the learned common feature subspace.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of array-like
            New samples, either stacked or given as a list of blocks.
        groups : array-like of shape (n_samples,), default=None
            Ignored for the common projection. Present for API consistency.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_common_components_)
            Common feature scores shared by all blocks.
        """
        check_is_fitted(self, "common_components_")
        if isinstance(X, (list, tuple)):
            blocks, _ = _check_multiblock_input(X)
            X_stacked = np.vstack(blocks)
        else:
            X_stacked = np.asarray(X, dtype=float)
            if X_stacked.ndim != 2:
                raise ValueError("`X` must be a 2D array or a list of 2D block arrays.")
        if X_stacked.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_stacked.shape[1]}.")
        return X_stacked @ self.common_components_

    def transform_individual(self, X, groups=None):
        """Project samples onto the individual feature subspaces.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of array-like
            New samples, either stacked or given as a list of blocks.
        groups : array-like of shape (n_samples,), default=None
            Block id for each sample when ``X`` is a single stacked matrix.

        Returns
        -------
        scores : list of ndarray
            One array per block, of shape (n_samples_in_block, individual_ranks_[i]).
        """
        check_is_fitted(self, "individual_components_")
        blocks, _ = _check_multiblock_input(X, groups)
        if len(blocks) != self.n_blocks_:
            raise ValueError(f"Expected {self.n_blocks_} blocks, got {len(blocks)}.")
        if any(block.shape[1] != self.n_features_in_ for block in blocks):
            raise ValueError(f"Expected {self.n_features_in_} features in every block, got mismatched blocks.")
        return [block @ components for block, components in zip(blocks, self.individual_components_)]
