import os
from urllib.request import urlretrieve

import h5py
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from .data_io import load_hdf5

HCP_LINK = {
    "REST1": "https://zenodo.org/records/10050233/files/HCP_BNA_intra_half_brain_REST1_Fisherz.hdf5",
    "REST2": "https://zenodo.org/records/10050233/files/HCP_BNA_intra_half_brain_REST2_Fisherz.hdf5",
}
GSP_LINK = "https://zenodo.org/records/10050234/files/gsp_BNA_intra_half_brain_Fisherz.mat"


def download_url_to_file(url, fpath):
    urlretrieve(url, fpath)


def split_functional_brain(matrix, connection_type="intra"):
    n_ = matrix.shape[1] / 2
    if connection_type == "intra":
        left = matrix[0::2, 0::2]
        right = matrix[1::2, 1::2]

        idx = np.triu_indices(n_, k=1)
        n_feat = int(n_ * (n_ - 1) / 2)
        left_vec = np.zeros((1, n_feat))
        right_vec = np.zeros((1, n_feat))
        left_vec[0, :] = left[idx]
        right_vec[0, :] = right[idx]
    elif connection_type == "inter":
        left = matrix[0::2, 1::2]
        right = matrix[1::2, 0::2]
        left_vec = left.reshape((1, -1))
        right_vec = right.reshape((1, -1))
    else:
        raise ValueError("Invalid connection type %s" % connection_type)

    return left_vec, right_vec


def save_half_brain(out_dir, out_fname, data_left, data_right):
    with h5py.File(os.path.join(out_dir, out_fname), "w") as f:
        f.create_dataset("Left", data=data_left)
        f.create_dataset("Right", data=data_right)


def save_half_brain_mat(out_dir, out_fname, data_left, data_right):
    savemat(os.path.join(out_dir, out_fname), {"Left": data_left, "Right": data_right})


def load_half_brain(
    data_dir,
    atlas,
    session=None,
    run=None,
    connection_type="intra",
    data_type="functional",
    dataset="HCP",
    download=True,
):
    if dataset == "HCP":
        if data_type == "functional":
            fname = "HCP_%s_%s_half_brain_%s_%s.hdf5" % (atlas, connection_type, session, run)
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                if download:
                    os.makedirs(data_dir, exist_ok=True)
                    print("Downloading %s session %s data, it may take 1-2 mins" % (dataset, session))
                    download_url_to_file(HCP_LINK[session], fpath)
                else:
                    raise ValueError("File %s does not exist" % fpath)
            data = load_hdf5(fpath)
        elif data_type == "structural":
            data = {"Left": [], "Right": []}
            fname = "%s_Volume.mat" % atlas
            data_in = loadmat(os.path.join(data_dir, fname))["%s_Volume" % atlas][0][0][0]
            data["Left"] = data_in[:, 0::2]
            data["Right"] = data_in[:, 1::2]
        else:
            raise ValueError("Invalid data type %s" % data_type)
    elif dataset in ["ABIDE", "ukb", "GSP"]:
        fpath = os.path.join(data_dir, "%s_%s_%s_half_brain_%s.mat" % (dataset, atlas, connection_type, run))
        if not os.path.exists(fpath):
            if download:
                if dataset == "GSP":
                    os.makedirs(data_dir, exist_ok=True)
                    print("Downloading %s data, it may take 1-2 mins." % dataset)
                    download_url_to_file(GSP_LINK, fpath)
                else:
                    raise ValueError("File %s does not exist" % fpath)
            else:
                raise ValueError("File %s does not exist" % fpath)
        data_file = loadmat(fpath)
        data = {"Left": data_file["Left"], "Right": data_file["Right"]}
    else:
        raise ValueError("Invalid dataset %s" % dataset)

    return data


def pick_half(data, random_state=144):
    x = np.zeros(data["Left"].shape)
    left_idx, right_idx = train_test_split(range(x.shape[0]), test_size=0.5, random_state=random_state)
    x[left_idx] = data["Left"][left_idx]
    x[right_idx] = data["Right"][right_idx]

    n_sub = x.shape[0]
    y = np.zeros(n_sub)
    y[left_idx] = 1
    y[right_idx] = -1

    x1 = np.zeros(data["Left"].shape)
    x1[left_idx] = data["Right"][left_idx]
    x1[right_idx] = data["Left"][right_idx]

    y1 = np.zeros(n_sub)
    y1[left_idx] = -1
    y1[right_idx] = 1

    y = label_binarize(y, classes=[-1, 1]).reshape(-1)
    y1 = label_binarize(y1, classes=[-1, 1]).reshape(-1)

    return x, y, x1, y1


def _pick_half_subs(data, random_state=144):
    n_ = data["Left"].shape[0]
    train_idx, _ = train_test_split(range(n_), test_size=0.5, random_state=random_state)
    x = np.concatenate([data["Left"][train_idx], data["Right"][train_idx]], axis=0)
    y = np.ones(n_)
    y[int(n_ / 2) :] = -1

    return x, y
