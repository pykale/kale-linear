import os

import numpy as np
import pytest
from numpy import testing
from scipy.io import loadmat
from tensorly.tenalg import multi_mode_dot

from kalelinear.transformer import MPCA

N_COMPS = [1, 50, 100]
VAR_RATIOS = [0.7, 0.95]
RELATIVE_TOL = 0.00001


@pytest.fixture(scope="module")
def baseline_model(download_path):
    baseline_data_path = os.path.join(download_path, "baseline.mat")
    if not os.path.exists(baseline_data_path):
        pytest.skip("Baseline data not found, skipping test_mpca_against_baseline.")
    return loadmat(baseline_data_path)


@pytest.mark.parametrize("n_components", N_COMPS)
@pytest.mark.parametrize("explained_variance_ratio", VAR_RATIOS)
def test_mpca(explained_variance_ratio, n_components, gait):
    # basic mpca test, return tensor
    X = gait["fea3D"].transpose((3, 0, 1, 2))
    mpca = MPCA(explained_variance_ratio=explained_variance_ratio, vectorize=False)
    X_proj = mpca.fit(X).transform(X)

    testing.assert_equal(X_proj.ndim, X.ndim)
    testing.assert_equal(X_proj.shape[0], X.shape[0])
    for i in range(1, X.ndim):
        assert X_proj.shape[i] <= X.shape[i]
        testing.assert_equal(mpca.proj_mats_[i - 1].shape[1], X.shape[i])

    X_rec = mpca.inverse_transform(X_proj)
    testing.assert_equal(X_rec.shape, X.shape)

    # test return vector
    mpca.set_params(**{"vectorize": True, "n_components": n_components})

    X_proj = mpca.transform(X)
    testing.assert_equal(X_proj.ndim, 2)
    testing.assert_equal(X_proj.shape[0], X.shape[0])
    testing.assert_equal(X_proj.shape[1], n_components)
    X_rec = mpca.inverse_transform(X_proj)
    testing.assert_equal(X_rec.shape, X.shape)

    # test n_samples = 1
    X0_proj = mpca.transform(X[0])
    testing.assert_equal(X0_proj.ndim, 2)
    testing.assert_equal(X0_proj.shape[0], 1)
    testing.assert_equal(X0_proj.shape[1], n_components)
    X0_rec = mpca.inverse_transform(X0_proj.reshape(-1))
    testing.assert_equal(X0_rec.shape[1:], X[0].shape)

    # test n_components exceeds upper limit
    mpca.set_params(**{"vectorize": True, "n_components": np.prod(X.shape[1:]) + 1})
    X_proj = mpca.transform(X)
    testing.assert_equal(X_proj.shape[1], np.prod(mpca.output_shape_))


def test_mpca_against_baseline(gait, baseline_model):
    X = gait["fea3D"].transpose((3, 0, 1, 2))
    baseline_proj_mats = [baseline_model["tUs"][i][0] for i in range(baseline_model["tUs"].size)]
    baseline_mean = baseline_model["TXmean"]
    mpca = MPCA(explained_variance_ratio=0.97)
    X_proj = mpca.fit(X).transform(X)
    testing.assert_allclose(baseline_mean, mpca.mean_)
    baseline_proj_X = multi_mode_dot(X - baseline_mean, baseline_proj_mats, modes=[1, 2, 3])
    # check whether the output embeddings is close to the baseline output by keeping the same variance ratio 97%
    testing.assert_allclose(X_proj**2, baseline_proj_X**2, rtol=RELATIVE_TOL)
    # testing.assert_equal(X_proj.shape, baseline_proj_X.shape)

    for i in range(X.ndim - 1):
        # check whether each eigen-vector column is equal to/opposite of corresponding baseline eigen-vector column
        # testing.assert_allclose(abs(mpca.proj_mats_[i]), abs(baseline_proj_mats[i]))
        testing.assert_allclose(mpca.proj_mats_[i] ** 2, baseline_proj_mats[i] ** 2, rtol=RELATIVE_TOL)


def test_transform_vectorize_override(gait):
    X = gait["fea3D"].transpose((3, 0, 1, 2))
    n_components = 50

    # init-level default and transform-level override to True
    mpca = MPCA(vectorize=False).fit(X)
    X_proj_tensor = mpca.transform(X)
    X_proj_vec = mpca.transform(X, vectorize=True)
    testing.assert_equal(X_proj_tensor.ndim, X.ndim)
    testing.assert_equal(X_proj_vec.ndim, 2)

    # init-level vectorize=True and transform-level override to False
    mpca = MPCA(vectorize=True, n_components=n_components).fit(X)
    X_proj_vec = mpca.transform(X)
    X_proj_tensor = mpca.transform(X, vectorize=False)
    testing.assert_equal(X_proj_vec.ndim, 2)
    testing.assert_equal(X_proj_vec.shape[1], n_components)
    testing.assert_equal(X_proj_tensor.ndim, X.ndim)

    # explicit None keeps the init-level setting
    X_proj_tensor_default = mpca.transform(X, vectorize=None)
    testing.assert_equal(X_proj_tensor_default.ndim, 2)


def test_fit_empty_input_raises():
    X = np.empty((0, 4, 5, 6))
    mpca = MPCA()
    with pytest.raises(ValueError, match="MPCA requires at least 2 samples to fit."):
        mpca.fit(X)
