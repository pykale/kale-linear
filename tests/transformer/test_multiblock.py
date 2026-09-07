import numpy as np
import pytest
from numpy import testing

from kalelinear.transformer import CIFE
from kalelinear.transformer._multiblock import _check_multiblock_input, _check_per_block_ranks
from tests.utils.test_utils import make_common_individual_dataset


@pytest.fixture(scope="module")
def multiblock_data():
    return make_common_individual_dataset(random_state=0)


def test_check_multiblock_input_rejects_groups_with_list():
    with pytest.raises(ValueError, match="`groups` must be None"):
        _check_multiblock_input([np.ones((5, 3)), np.ones((5, 3))], groups=np.zeros(10))


def test_check_multiblock_input_accepts_nested_list_with_groups():
    X = np.arange(20, dtype=float).reshape(10, 2)
    groups = np.repeat([0, 1], 5)
    expected_blocks, expected_groups, expected_ids = _check_multiblock_input(X, groups)
    blocks, out_groups, block_ids = _check_multiblock_input(X.tolist(), groups)
    for actual, expected in zip(blocks, expected_blocks):
        testing.assert_allclose(actual, expected)
    testing.assert_array_equal(out_groups, expected_groups)
    testing.assert_array_equal(block_ids, expected_ids)


def test_check_multiblock_input_treats_nested_list_without_groups_as_stacked():
    with pytest.raises(ValueError, match="`groups` must be provided"):
        _check_multiblock_input([[1.0, 2.0], [3.0, 4.0]])


def test_check_multiblock_input_rejects_empty_list():
    with pytest.raises(ValueError, match="at least one block"):
        _check_multiblock_input([])


def test_check_multiblock_input_rejects_1d_block():
    with pytest.raises(ValueError, match="2D array"):
        _check_multiblock_input([np.ones(5), np.ones((5, 3))])


def test_check_multiblock_input_rejects_empty_block():
    with pytest.raises(ValueError, match="at least one sample"):
        _check_multiblock_input([np.ones((0, 3)), np.ones((5, 3))])


def test_check_multiblock_input_rejects_mismatched_features():
    with pytest.raises(ValueError, match="same number of features"):
        _check_multiblock_input([np.ones((5, 3)), np.ones((5, 4))])


def test_check_multiblock_input_rejects_single_block_list():
    with pytest.raises(ValueError, match="(?i)at least two blocks"):
        _check_multiblock_input([np.ones((5, 3))])


def test_check_multiblock_input_accepts_single_block_with_min_blocks_one():
    blocks, groups, block_ids = _check_multiblock_input([np.ones((5, 3))], min_blocks=1)
    assert len(blocks) == 1
    assert groups is None
    testing.assert_array_equal(block_ids, [0])


def test_check_multiblock_input_rejects_1d_stacked():
    with pytest.raises(ValueError, match="2D array or a list"):
        _check_multiblock_input(np.ones(5))


def test_check_multiblock_input_requires_groups():
    with pytest.raises(ValueError, match="`groups` must be provided"):
        _check_multiblock_input(np.ones((10, 3)))


def test_check_multiblock_input_validates_groups_shape():
    X = np.ones((10, 3))
    with pytest.raises(ValueError, match="1D array aligned"):
        _check_multiblock_input(X, groups=np.ones((10, 1)))
    with pytest.raises(ValueError, match="1D array aligned"):
        _check_multiblock_input(X, groups=np.zeros(5))


def test_check_multiblock_input_rejects_single_group():
    with pytest.raises(ValueError, match="(?i)at least 2 blocks"):
        _check_multiblock_input(np.ones((10, 3)), groups=np.zeros(10, dtype=int))


def test_check_multiblock_input_rejects_nan_group_block():
    X = np.ones((6, 3))
    groups = np.array([0.0, 0.0, 1.0, 1.0, np.nan, np.nan])
    with pytest.raises(ValueError, match="at least one sample"):
        _check_multiblock_input(X, groups)


def test_check_multiblock_input_orders_blocks_by_first_appearance():
    X = np.arange(6).reshape(6, 1)
    groups = np.array([2, 2, 0, 0, 1, 1])
    blocks, _, block_ids = _check_multiblock_input(X, groups)
    testing.assert_array_equal([block[0, 0] for block in blocks], [0.0, 2.0, 4.0])
    testing.assert_array_equal(block_ids, [2, 0, 1])


def test_check_per_block_ranks_accepts_integer():
    testing.assert_array_equal(_check_per_block_ranks(3, 4, "ranks"), np.full(4, 3, dtype=int))


def test_check_per_block_ranks_accepts_sequence():
    testing.assert_array_equal(_check_per_block_ranks([1, 2], 2, "ranks"), np.array([1, 2]))


def test_check_per_block_ranks_accepts_none():
    assert _check_per_block_ranks(None, 3, "ranks") is None


def test_check_per_block_ranks_rejects_wrong_length():
    with pytest.raises(ValueError, match="one value per block"):
        _check_per_block_ranks([1, 2], 3, "ranks")


def test_check_per_block_ranks_rejects_non_numeric():
    with pytest.raises(ValueError, match="numeric"):
        _check_per_block_ranks(["a", "b"], 2, "ranks")


def test_check_per_block_ranks_rejects_complex():
    with pytest.raises(ValueError, match="real numeric"):
        _check_per_block_ranks([1 + 1j, 2], 2, "ranks")


def test_check_per_block_ranks_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        _check_per_block_ranks([1.0, np.nan], 2, "ranks")


def test_check_per_block_ranks_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        _check_per_block_ranks([1, -1], 2, "ranks")


def test_check_per_block_ranks_rejects_non_integer():
    with pytest.raises(ValueError, match="integer"):
        _check_per_block_ranks([1.5, 2.0], 2, "ranks")


def test_transform_rejects_1d_input(multiblock_data):
    X, groups, _, _ = multiblock_data
    cife = CIFE(n_common_components=2, random_state=0).fit(X, groups=groups)
    with pytest.raises(ValueError, match="2D array or a list"):
        cife.transform(np.ones(X.shape[1]))


def test_transform_rejects_wrong_feature_count(multiblock_data):
    X, groups, _, _ = multiblock_data
    cife = CIFE(n_common_components=2, random_state=0).fit(X, groups=groups)
    with pytest.raises(ValueError, match="Expected .* features, got"):
        cife.transform(np.ones((5, X.shape[1] + 1)))


def test_transform_individual_rejects_wrong_block_count(multiblock_data):
    X, groups, _, _ = multiblock_data
    cife = CIFE(n_common_components=2, random_state=0).fit(X, groups=groups)
    with pytest.raises(ValueError, match="Expected 3 blocks, got 2"):
        cife.transform_individual([X[:5], X[5:10]])


def test_transform_individual_rejects_wrong_feature_count(multiblock_data):
    X, groups, _, _ = multiblock_data
    cife = CIFE(n_common_components=2, random_state=0).fit(X, groups=groups)
    n_features = X.shape[1]
    blocks = [np.ones((5, n_features + 1)) for _ in range(3)]
    with pytest.raises(ValueError, match="features in every block"):
        cife.transform_individual(blocks)
