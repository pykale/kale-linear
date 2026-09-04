import numpy as np
import pytest
from numpy import testing
from sklearn.base import clone

from kalelinear.transformer import AJIVE
from kalelinear.transformer._ajive import _jive_rand_null_norm
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


def test_ajive_automatic_initial_ranks_are_capped_at_numerical_rank(multiblock_data):
    X, groups, blocks, _ = multiblock_data
    # variance_threshold=1.0 must not count machine-noise singular values: the
    # planted blocks have numerical ranks 5, 6 and 4, not the full 30 columns.
    ranks = AJIVE(variance_threshold=1.0)._resolve_initial_ranks(blocks)
    testing.assert_array_equal(ranks, [5, 6, 4])


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


def test_ajive_transform_individual_rejects_unknown_block_ids(multiblock_data):
    X, groups, _, _ = multiblock_data
    ajive = AJIVE(n_resamples=50, random_state=0)
    ajive.fit(X, groups=groups)
    with pytest.raises(ValueError, match="block ids"):
        ajive.transform_individual(X, groups=groups + 10)


def test_ajive_transform_individual_preserves_block_assignment(multiblock_data):
    X, groups, _, _ = multiblock_data
    ajive = AJIVE(n_resamples=50, random_state=0)
    ajive.fit(X, groups=groups)
    expected = ajive.transform_individual(X, groups=groups)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    actual = ajive.transform_individual(X[perm], groups=groups[perm])
    position = np.empty(len(perm), dtype=int)
    position[perm] = np.arange(len(perm))
    for k, block_id in enumerate(np.unique(groups)):
        row_indices = np.flatnonzero(groups == block_id)
        row_order = np.argsort(position[row_indices])
        testing.assert_allclose(actual[k], expected[k][row_order])


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


def test_ajive_zero_energy_block_is_rejected():
    blocks = [np.zeros((5, 3)), np.ones((5, 3))]
    with pytest.raises(ValueError, match="positive values"):
        AJIVE(n_resamples=50, random_state=0).fit(blocks)


def test_ajive_rejects_zero_energy_block_with_explicit_initial_ranks():
    blocks = [np.zeros((5, 3)), np.ones((5, 3))]
    with pytest.raises(ValueError, match="zero energy"):
        AJIVE(initial_ranks=[2, 1], n_resamples=20, random_state=0).fit(blocks)


def test_jive_rand_null_norm_limits_directions_to_null_space_dimension():
    rng = np.random.RandomState(0)
    n_ambient, rank = 30, 20  # the null space has only 10 dimensions
    data = rng.randn(60, n_ambient)
    basis, _ = np.linalg.qr(rng.randn(n_ambient, rank))

    null_norms = _jive_rand_null_norm(data, basis, 5, rng)
    # With the direction count capped at the null-space dimension, the sampled
    # directions span the entire null space and the spectral norm becomes the
    # exact operator norm of the data restricted to that null space.
    expected = np.linalg.norm(data - data @ basis @ basis.T, ord=2)
    testing.assert_allclose(null_norms, expected, rtol=1e-8)


def test_ajive_accepts_initial_ranks_larger_than_null_space():
    rng = np.random.RandomState(0)
    # Full-rank 40 x 8 blocks: requesting 6 components leaves only a
    # 2-dimensional null space, i.e. fewer orthogonal null directions than the
    # requested rank. The fit must cap the sampled directions instead of
    # normalising round-off noise, while the ranks stay below the numerical
    # rank of each block.
    blocks = [rng.randn(40, 8) for _ in range(3)]
    ajive = AJIVE(initial_ranks=[6, 6, 6], n_resamples=20, random_state=0)
    ajive.fit(blocks)
    assert ajive.common_components_.shape == (8, ajive.n_common_components_)
    assert np.all(np.isfinite(ajive.common_components_))


def test_ajive_common_rank_is_capped_by_smallest_initial_rank():
    rng = np.random.RandomState(0)
    D = 10
    # e2 is shared by blocks 1-2 only; the joint space shared by every block
    # is span{e1}, so at most min(initial_ranks) = 1 component can be common.
    V1 = np.eye(D)[[0, 1]].T
    V2 = np.eye(D)[[0, 1]].T
    V3 = np.eye(D)[[0]].T
    blocks = [rng.randn(40, V.shape[1]) @ V.T for V in (V1, V2, V3)]
    ajive = AJIVE(
        n_common_components=5,  # more than the theoretical joint-space dimension
        initial_ranks=[2, 2, 1],
        n_resamples=50,
        random_state=0,
    )
    ajive.fit(blocks)
    assert ajive.n_common_components_ <= 1
    assert ajive.common_components_.shape[1] <= 1


def test_ajive_is_invariant_to_data_scaling(multiblock_data):
    X, groups, blocks, _ = multiblock_data
    baseline = AJIVE(initial_ranks=[5, 6, 4], n_resamples=50, random_state=0)
    baseline.fit(blocks)
    for scale in (1e-12, 1e12):
        scaled = AJIVE(initial_ranks=[5, 6, 4], n_resamples=50, random_state=0)
        scaled.fit([block * scale for block in blocks])
        assert scaled.n_common_components_ == baseline.n_common_components_
        testing.assert_array_equal(scaled.individual_ranks_, baseline.individual_ranks_)


def test_ajive_rejects_initial_ranks_exceeding_numerical_rank():
    rng = np.random.RandomState(0)
    # The first block has an exactly-zero column, so its numerical rank is 4
    # even though it is 40 x 5. Requesting rank 5 makes the smallest retained
    # singular value zero and the Wedin angle bound undefined.
    blocks = [np.column_stack([rng.randn(40, 4), np.zeros(40)]), rng.randn(40, 5), rng.randn(40, 5)]
    with pytest.raises(ValueError, match="numerical rank"):
        AJIVE(initial_ranks=[5, 2, 2], n_resamples=20, random_state=0).fit(blocks)


def test_ajive_rejects_invalid_per_block_rank_specs(multiblock_data):
    X, groups, _, _ = multiblock_data
    invalid_specs = [
        ([np.nan, 2, 2], "NaN"),
        ([np.inf, 2, 2], "infinite"),
        ([1.5, 2, 2], "integer"),
        ([-1, 2, 2], "non-negative"),
    ]
    for spec, message in invalid_specs:
        with pytest.raises(ValueError, match=message):
            AJIVE(n_individual_components=spec, n_resamples=20, random_state=0).fit(X, groups=groups)
    with pytest.raises(ValueError, match="NaN"):
        AJIVE(initial_ranks=[np.nan, 2, 2], n_resamples=20, random_state=0).fit(X, groups=groups)


def test_ajive_transform_individual_rejects_wrong_number_of_blocks(multiblock_data):
    X, groups, blocks, _ = multiblock_data
    ajive = AJIVE(n_resamples=20, random_state=0).fit(X, groups=groups)
    with pytest.raises(ValueError, match="Expected 3 blocks"):
        ajive.transform_individual(blocks[:2])
    with pytest.raises(ValueError, match="Expected 3 blocks"):
        ajive.transform_individual(blocks + [blocks[0]])
