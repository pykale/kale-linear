import os

import h5py
import numpy as np
import pandas as pd


def load_txt(fpaths, connection_type="intra"):
    """Load left/right half-brain vectors from text connectivity matrices."""
    from .half_brain import split_functional_brain

    left_ = []
    right_ = []
    for fpath in fpaths:
        data_matrix = np.genfromtxt(fpath)
        left_vec, right_vec = split_functional_brain(data_matrix, connection_type=connection_type)
        left_.append(left_vec.reshape((1, -1)))
        right_.append(right_vec.reshape((1, -1)))

    return np.concatenate(left_, axis=0), np.concatenate(right_, axis=0)


def load_hdf5(fpath):
    with h5py.File(fpath, "r") as f:
        return {"Left": f["Left"][()], "Right": f["Right"][()]}


def read_tabular(fname, **kwargs):
    """Read a table from a .xlsx or .csv file."""
    file_format = fname.split(".")[-1]
    if file_format == "xlsx":
        return pd.read_excel(fname, engine="openpyxl", **kwargs)
    if file_format == "csv":
        return pd.read_csv(fname, **kwargs)

    raise ValueError("Unsupported file type %s" % file_format)


def get_fpaths(fdir, idx_list, file_format="txt"):
    """Return existing subject-indexed files under a directory."""
    fpaths = {}
    for idx in idx_list:
        fname = "%s.%s" % (idx, file_format)
        fpath = os.path.join(fdir, fname)
        if os.path.exists(fpath):
            fpaths[idx] = fpath

    return pd.DataFrame(data={"File path": fpaths.values()}, index=list(fpaths.keys()))


def file_split(file_path):
    filepath, tempfilename = os.path.split(file_path)
    filename, extension = os.path.splitext(tempfilename)
    return filepath, filename, extension


def read_file(file):
    _, _, ext = file_split(file)
    if ext == ".h5":
        return pd.read_hdf(file)
    if ext == ".pkl":
        return pd.read_pickle(file)

    raise ValueError("Unsupported file type %s, supported type: xxx.h5 or xxx.pkl..." % ext)


def select_data_subj(df, subj_id, df_column_name):
    return df[df_column_name][df["subject"] == subj_id]


def select_data_multi_subj(df, subj_ids, df_column_name):
    subj_df = df["subject"].to_numpy()
    _, index1, _ = np.intersect1d(subj_df, subj_ids, assume_unique=False, return_indices=True)
    return df.loc[index1, df_column_name].to_numpy()
