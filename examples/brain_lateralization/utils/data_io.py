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


def load_result(dataset, root_dir, lambdas, seed_start, test_size=0.0):
    """load brain left/right classification results for a dataset

    Args:
        dataset (string): _description_
        root_dir (string): _description_
        lambdas (list): _description_
        seed_start (_type_): _description_
        test_size (float, optional): _description_. Defaults to 0.0.
    """
    res_dict = dict()
    res_list = []
    test_size_str = str(int(test_size * 10))
    for lambda_ in lambdas:
        res_dict[lambda_] = []

    for lambda_ in lambdas:
        if not isinstance(lambda_, str):
            lambda_str = str(int(lambda_))
        else:
            lambda_str = lambda_
        model_dir = os.path.join(root_dir, "lambda%s" % lambda_str)
        for seed_iter in range(51):
            random_state = seed_start - seed_iter
            res_fname = "results_%s_L%s_test_size0%s_Fisherz_%s.csv" % (
                dataset,
                lambda_str,
                test_size_str,
                random_state,
            )
            res_fpath = os.path.join(model_dir, res_fname)
            if os.path.exists(res_fpath):
                res_df = pd.read_csv(os.path.join(model_dir, res_fname))
                res_df["seed"] = random_state
                res_dict[lambda_].append(res_df)
                res_list.append(res_df)

    for lambda_ in lambdas:
        res_dict[lambda_] = pd.concat(res_dict[lambda_])

    res_df_all = pd.concat(res_list)
    res_df_all = res_df_all.reset_index(drop=True)

    return res_df_all


def reformat_results(res_df, test_sets, male_label=0):
    """reformat results dataframe to one accuracy per row

    Args:
        res_df (_type_): _description_
        test_sets (_type_): _description_
        male_label (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    res_reformat = {
        "Accuracy": [],
        "Test set": [],
        "Lambda": [],
        "Target group": [],
        "seed": [],
        "split": [],
        "fold": [],
        "Train session": [],
    }
    for idx_ in res_df.index:
        # print(idx_, res_df.iloc[idx_, 12])
        subset_ = res_df.loc[idx_, :]
        for test_set in test_sets:
            res_reformat["Accuracy"].append(subset_[test_set])
            res_reformat["Lambda"].append(subset_["lambda"])
            if "train_session" in subset_:
                res_reformat["Train session"].append(subset_["train_session"])
            else:
                res_reformat["Train session"].append(None)
            if "target_group" not in subset_:  # for GSP dataset
                _group = subset_["train_gender"]
            else:
                _group = subset_["target_group"]
            test_set_list = test_set.split("_")
            if _group == "Male" or _group == male_label:
                res_reformat["Target group"].append("Male")
                if "oc" in test_set_list or "tgt" in test_set_list:
                    res_reformat["Test set"].append("Female")
                else:
                    res_reformat["Test set"].append("Male")
            else:
                res_reformat["Target group"].append("Female")
                if "oc" in test_set_list or "tgt" in test_set_list:
                    res_reformat["Test set"].append("Male")
                else:
                    res_reformat["Test set"].append("Female")

            for key in ["seed", "split", "fold"]:
                res_reformat[key].append(subset_[key])
    return pd.DataFrame(res_reformat)
