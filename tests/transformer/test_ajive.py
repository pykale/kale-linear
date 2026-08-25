import numpy as np
import pytest
from numpy import testing
from sklearn.base import clone

from kalelinear.transformer import AJIVE
from tests.utils.test_utils import make_common_individual_dataset


@pytest.fixture(scope="module")
def multiblock_data():
    return make_common_individual_dataset(random_state=0)


def _subspace_error(estimated, planted):
    projection = planted @ planted.T
    return np.linalg.norm(estimated - projection @ estimated) / np.sqrt(estimated.shape[1])


def test_ajive_recovers_joint_and_individual_subspaces(multiblock_data):
    X, groups, _, common_basis = multiblock_data
    ajive = AJIVE(initial_ranks=[5, 6, 4], n_resamples=50, random_state=0)
    ajive.fit(X, groups=groups)

    assert ajive.n_common_components_ == 2
    assert ajive.common_components_.shape == (X.shape[1], 2)
    assert _subspace_error(ajive.common_components_, common_basis) < 1e-6
    testing.assert_array_equal(ajive.individual_ranks_, [3, 4, 2])


def test_ajive_automatic_initial_ranks(multiblock_data):
    X, groups, _, _ = multiblock_data
    ajive = AJIVE(variance_threshold=1.0, n_resamples=50, random_state=0)
    ajive.fit(X, groups=groups)
    assert ajive.n_common_components_ == 2


def test_ajive_percentile_uses_larger_perturbation_bound(multiblock_data):
    X, groups, _, _ = multiblock_data
    ranks = [
        AJIVE(percentile=percentile, n_resamples=50, random_state=0).fit(X, groups=groups).n_common_components_
        for percentile in (5, 50, 95)
    ]
    # A larger percentile gives a non-decreasing Wedin threshold, so the joint
    # rank selected from max(wedin, random-direction) bound cannot increase.
    assert ranks[0] >= ranks[1] >= ranks[2]


def test_ajive_manual_joint_rank(multiblock_data):
    X, groups, _, _ = multiblock_data
    ajive = AJIVE(
        n_common_components=1,
        initial_ranks=[5, 6, 4],
        n_resamples=50,
        random_state=0,
    )
    ajive.fit(X, groups=groups)
    assert ajive.n_common_components_ == 1


def test_ajive_individual_rank_override(multiblock_data):
    X, groups, _, _ = multiblock_data
    ajive = AJIVE(
        initial_ranks=[5, 6, 4],
        n_individual_components=[2, 2, 2],
        n_resamples=50,
        random_state=0,
    )
    ajive.fit(X, groups=groups)
    testing.assert_array_equal(ajive.individual_ranks_, [2, 2, 2])


def test_ajive_list_input_matches_stacked(multiblock_data):
    X, groups, blocks, _ = multiblock_data
    from_stacked = AJIVE(n_resamples=50, random_state=0).fit(X, groups=groups)
    from_list = AJIVE(n_resamples=50, random_state=0).fit(blocks)
    testing.assert_allclose(from_stacked.common_components_, from_list.common_components_)


def test_ajive_transform_consistency(multiblock_data):
    X, groups, blocks, _ = multiblock_data
    ajive = AJIVE(n_resamples=50, random_state=0)
    ajive.fit(X, groups=groups)

    z = ajive.transform(X)
    testing.assert_allclose(z, X @ ajive.common_components_)
    testing.assert_allclose(ajive.transform(blocks), z)
    single_block = ajive.transform([blocks[0]])
    testing.assert_allclose(single_block, blocks[0] @ ajive.common_components_)
    testing.assert_allclose(ajive.transform((blocks[0],)), single_block)

    individual = ajive.transform_individual(X, groups=groups)
    assert [scores.shape for scores in individual] == [(60, 3), (50, 4), (70, 2)]
    individual_from_blocks = ajive.transform_individual(blocks)
    for expected, actual in zip(individual, individual_from_blocks):
        testing.assert_allclose(expected, actual)


def test_ajive_fit_transform_and_clone(multiblock_data):
    X, groups, _, _ = multiblock_data
    ajive = AJIVE(n_resamples=50, random_state=0)
    z = ajive.fit_transform(X, groups=groups)
    assert z.shape == (X.shape[0], 2)

    copied = clone(ajive)
    copied.fit(X, groups=groups)
    testing.assert_allclose(copied.common_components_, ajive.common_components_)


def test_ajive_feature_names(multiblock_data):
    X, groups, _, _ = multiblock_data
    ajive = AJIVE(n_resamples=50, random_state=0).fit(X, groups=groups)
    testing.assert_array_equal(ajive.get_feature_names_out(), np.array(["ajive0", "ajive1"]))


def test_ajive_input_validation():
    X = np.ones((10, 4))
    with pytest.raises(ValueError, match="`groups` must be provided"):
        AJIVE().fit(X)
    with pytest.raises(ValueError, match="same number of features"):
        AJIVE().fit([np.ones((5, 3)), np.ones((5, 4))])
    with pytest.raises(ValueError, match="initial_ranks"):
        AJIVE(initial_ranks=[5, 6]).fit([np.ones((4, 3)), np.ones((4, 3))])
