import numpy as np
import pytest
from numpy import testing
from sklearn.base import clone

from kalelinear.transformer import CIFE
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


def test_cife_transform_consistency(multiblock_data):
    X, groups, blocks, _ = multiblock_data
    cife = CIFE(n_common_components=2, n_individual_components=[3, 4, 2], random_state=0)
    cife.fit(X, groups=groups)

    z = cife.transform(X)
    testing.assert_allclose(z, X @ cife.common_components_)
    testing.assert_allclose(cife.transform(blocks), z)

    individual = cife.transform_individual(X, groups=groups)
    assert [scores.shape for scores in individual] == [(60, 3), (50, 4), (70, 2)]
    individual_from_blocks = cife.transform_individual(blocks)
    for expected, actual in zip(individual, individual_from_blocks):
        testing.assert_allclose(expected, actual)


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


def test_cife_input_validation():
    X = np.ones((10, 4))
    with pytest.raises(ValueError, match="`groups` must be provided"):
        CIFE().fit(X)
    with pytest.raises(ValueError, match="same number of features"):
        CIFE().fit([np.ones((5, 3)), np.ones((5, 4))])
    with pytest.raises(ValueError, match="(?i)at least two blocks"):
        CIFE().fit([np.ones((5, 3))])
