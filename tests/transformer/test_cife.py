import numpy as np
import pytest
from numpy import testing
from sklearn.base import clone

from kalelinear.transformer import CIFE
from kalelinear.transformer._cife import _column_space_basis
from tests.utils.test_utils import make_common_individual_dataset


@pytest.fixture(scope="module")
def multiblock_data():
    return make_common_individual_dataset(random_state=0)


def _subspace_error(estimated, planted):
    projection = planted @ planted.T
    return np.linalg.norm(estimated - projection @ estimated) / np.sqrt(estimated.shape[1])


def test_cife_recovers_common_and_individual_subspaces(multiblock_data):
    X, groups, _, common_basis = multiblock_data
    cife = CIFE(
        n_common_components=2,
        n_individual_components=[3, 4, 2],
        tol=1e-10,
        random_state=0,
    )
    cife.fit(X, groups=groups)

    assert cife.n_common_components_ == 2
    assert cife.common_components_.shape == (X.shape[1], 2)
    assert _subspace_error(cife.common_components_, common_basis) < 1e-4
    testing.assert_array_equal(cife.individual_ranks_, [3, 4, 2])
    assert [components.shape[1] for components in cife.individual_components_] == [3, 4, 2]


def test_cife_automatic_common_rank(multiblock_data):
    X, groups, _, _ = multiblock_data
    cife = CIFE(tol=1e-10, random_state=0)
    cife.fit(X, groups=groups)
    assert cife.n_common_components_ == 2


def test_cife_detects_no_common_structure():
    random_state = np.random.RandomState(1)
    blocks = [random_state.randn(50, 4) @ random_state.randn(30, 4).T for _ in range(3)]
    cife = CIFE(tol=1e-10, random_state=0)
    cife.fit(blocks)
    assert cife.n_common_components_ == 0
    assert cife.common_components_.shape == (30, 0)


def test_cife_list_input_matches_stacked(multiblock_data):
    X, groups, blocks, _ = multiblock_data
    from_stacked = CIFE(tol=1e-10, random_state=0).fit(X, groups=groups)
    from_list = CIFE(tol=1e-10, random_state=0).fit(blocks)
    testing.assert_allclose(from_stacked.common_components_, from_list.common_components_)


def test_cife_accepts_stacked_python_list(multiblock_data):
    X, groups, _, _ = multiblock_data
    cife = CIFE(n_common_components=2, n_individual_components=[3, 4, 2], random_state=0)
    cife.fit(X.tolist(), groups=groups)

    testing.assert_allclose(cife.transform(X.tolist()), X @ cife.common_components_)
    expected_individual = cife.transform_individual(X, groups=groups)
    actual_individual = cife.transform_individual(X.tolist(), groups=groups)
    for expected, actual in zip(expected_individual, actual_individual):
        testing.assert_allclose(expected, actual)


def test_cife_fit_rejects_unexpected_keyword(multiblock_data):
    X, groups, _, _ = multiblock_data
    with pytest.raises(TypeError, match="unexpected keyword argument 'group'"):
        CIFE(random_state=0).fit(X, group=groups)


def test_cife_transform_consistency(multiblock_data):
    X, groups, blocks, _ = multiblock_data
    cife = CIFE(n_common_components=2, n_individual_components=[3, 4, 2], random_state=0)
    cife.fit(X, groups=groups)

    z = cife.transform(X)
    testing.assert_allclose(z, X @ cife.common_components_)
    testing.assert_allclose(cife.transform(blocks), z)
    single_block = cife.transform([blocks[0]])
    testing.assert_allclose(single_block, blocks[0] @ cife.common_components_)
    testing.assert_allclose(cife.transform((blocks[0],)), single_block)

    individual = cife.transform_individual(X, groups=groups)
    assert [scores.shape for scores in individual] == [(60, 3), (50, 4), (70, 2)]
    individual_from_blocks = cife.transform_individual(blocks)
    for expected, actual in zip(individual, individual_from_blocks):
        testing.assert_allclose(expected, actual)


def test_cife_transform_individual_rejects_unknown_block_ids(multiblock_data):
    X, groups, _, _ = multiblock_data
    cife = CIFE(n_common_components=2, n_individual_components=[3, 4, 2], random_state=0)
    cife.fit(X, groups=groups)
    with pytest.raises(ValueError, match="block ids"):
        cife.transform_individual(X, groups=groups + 10)


def test_cife_transform_individual_preserves_block_assignment(multiblock_data):
    X, groups, _, _ = multiblock_data
    cife = CIFE(n_common_components=2, n_individual_components=[3, 4, 2], random_state=0)
    cife.fit(X, groups=groups)
    expected = cife.transform_individual(X, groups=groups)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    actual = cife.transform_individual(X[perm], groups=groups[perm])
    position = np.empty(len(perm), dtype=int)
    position[perm] = np.arange(len(perm))
    for k, block_id in enumerate(np.unique(groups)):
        row_indices = np.flatnonzero(groups == block_id)
        row_order = np.argsort(position[row_indices])
        testing.assert_allclose(actual[k], expected[k][row_order])


def test_cife_fit_transform_and_clone(multiblock_data):
    X, groups, _, _ = multiblock_data
    cife = CIFE(n_common_components=2, random_state=0)
    z = cife.fit_transform(X, groups=groups)
    assert z.shape == (X.shape[0], 2)

    copied = clone(cife)
    copied.fit(X, groups=groups)
    testing.assert_allclose(copied.common_components_, cife.common_components_)


def test_cife_feature_names(multiblock_data):
    X, groups, _, _ = multiblock_data
    cife = CIFE(n_common_components=2, random_state=0).fit(X, groups=groups)
    testing.assert_array_equal(cife.get_feature_names_out(), np.array(["cife0", "cife1"]))


def test_cife_full_rank_block_requires_pca_dim():
    random_state = np.random.RandomState(2)
    blocks = [random_state.randn(40, 10) for _ in range(3)]
    with pytest.raises(ValueError, match="whole feature space"):
        CIFE(random_state=0).fit(blocks)
    cife = CIFE(pca_dim=0.5, random_state=0)
    cife.fit(blocks)
    assert cife.n_common_components_ == 0


def test_cife_pca_dim_truncates_rank_deficient_blocks():
    rng = np.random.RandomState(0)
    # Each block has 30 features but only 20 numerical dimensions.
    low_rank = rng.randn(30, 20) @ rng.randn(20, 60)
    blocks = [low_rank.T for _ in range(3)]
    cife = CIFE(n_common_components=8, pca_dim=5, random_state=0)
    cife.fit(blocks)
    assert cife.n_common_components_ <= 5


def test_column_space_basis_applies_pca_dim_to_rank_deficient_block():
    rng = np.random.RandomState(0)
    Y = rng.randn(30, 20) @ rng.randn(20, 60)
    assert _column_space_basis(Y, None)[1] == 20
    assert _column_space_basis(Y, 5)[1] == 5


def test_cife_zero_common_components_skips_full_rank_validation():
    random_state = np.random.RandomState(2)
    blocks = [random_state.randn(40, 10) for _ in range(3)]
    cife = CIFE(n_common_components=0, random_state=0)
    cife.fit(blocks)
    assert cife.n_common_components_ == 0
    assert cife.common_components_.shape == (10, 0)
    testing.assert_array_equal(cife.individual_ranks_, [10, 10, 10])


def test_cife_rejects_infinite_individual_ranks(multiblock_data):
    X, groups, _, _ = multiblock_data
    with pytest.raises(ValueError, match="infinite"):
        CIFE(n_individual_components=[np.inf, 3, 2], random_state=0).fit(X, groups=groups)


def test_cife_rejects_invalid_per_block_rank_specs(multiblock_data):
    X, groups, _, _ = multiblock_data
    invalid_specs = [
        ([np.nan, 3, 2], "NaN"),
        ([1.5, 3, 2], "integer"),
        ([-1, 3, 2], "non-negative"),
        ([1 + 1j, 3, 2], "real numeric"),
    ]
    for spec, message in invalid_specs:
        with pytest.raises(ValueError, match=message):
            CIFE(n_individual_components=spec, random_state=0).fit(X, groups=groups)


def test_cife_validates_tol_and_epsilon_are_in_half_open_unit_interval(multiblock_data):
    X, groups, _, _ = multiblock_data
    for kwargs in ({"tol": 1.0}, {"tol": 1.5}, {"tol": -0.1}, {"epsilon": 1.0}, {"epsilon": 1.5}, {"epsilon": -0.1}):
        with pytest.raises(ValueError, match="must be a float in the range \\[0.0, 1.0\\)"):
            CIFE(**kwargs).fit(X, groups=groups)


def test_cife_validates_pca_dim_domain():
    rng = np.random.RandomState(2)
    full_rank_blocks = [rng.randn(40, 10) for _ in range(3)]
    for pca_dim in (0.0, 1.0):
        with pytest.raises(ValueError, match="pca_dim"):
            CIFE(pca_dim=pca_dim, random_state=0).fit(full_rank_blocks)
    # Fractions are strictly inside (0, 1); component counts are integers >= 1.
    for pca_dim in (0.5, 1, 2):
        CIFE(pca_dim=pca_dim, random_state=0).fit(full_rank_blocks)


def test_cife_transform_individual_rejects_wrong_number_of_blocks(multiblock_data):
    X, groups, blocks, _ = multiblock_data
    cife = CIFE(n_common_components=2, n_individual_components=[3, 4, 2], random_state=0)
    cife.fit(X, groups=groups)
    with pytest.raises(ValueError, match="Expected 3 blocks"):
        cife.transform_individual(blocks[:2])


def test_cife_input_validation():
    X = np.ones((10, 4))
    with pytest.raises(ValueError, match="`groups` must be provided"):
        CIFE().fit(X)
    with pytest.raises(ValueError, match="same number of features"):
        CIFE().fit([np.ones((5, 3)), np.ones((5, 4))])
    with pytest.raises(ValueError, match="(?i)at least two blocks"):
        CIFE().fit([np.ones((5, 3))])
