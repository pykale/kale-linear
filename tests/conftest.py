# Global settings for tests. Run before any test
import os

import pytest
from scipy.io import loadmat


@pytest.fixture(scope="session")
def download_path():
    path = os.path.join("tests", "test_data")
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="module")
def gait(download_path):
    gait_data_path = os.path.join(download_path, "gait.mat")
    if not os.path.exists(gait_data_path):
        pytest.skip("Gait data not found, skipping tests.")
    return loadmat(gait_data_path)
