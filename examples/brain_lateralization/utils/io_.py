import os

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.hub import download_url_to_file

HCP_LINK = {
    "REST1": "https://zenodo.org/records/10050233/files/HCP_BNA_intra_half_brain_REST1_Fisherz.hdf5",
    "REST2": "https://zenodo.org/records/10050233/files/HCP_BNA_intra_half_brain_REST2_Fisherz.hdf5",
}
GSP_LINK = "https://zenodo.org/records/10050234/files/gsp_BNA_intra_half_brain_Fisherz.mat"


def load_txt(fpaths, connection_type="intra"):
    """Load data from a list of txt files。

    Parameters
    ----------
    fpaths (list): A list of .txt file paths.

    Returns
    -------
        Two ndarrays
    """
    left_ = []
    right_ = []
    for fpath in fpaths:
        data_matrix = np.genfromtxt(fpath)
        left_vec, right_vec = split_functional_brain(data_matrix, connection_type=connection_type)
        left_.append(left_vec.reshape((1, -1)))
        right_.append(right_vec.reshape((1, -1)))
    left_ = np.concatenate(left_, axis=0)
    right_ = np.concatenate(right_, axis=0)

    return left_, right_


def load_hdf5(fpath):
    """

    Parameters
    ----------
    fpath

    Returns
    -------

    """
    f = h5py.File(fpath, "r")
    data = {"Left": f["Left"][()], "Right": f["Right"][()]}
    return data


def read_tabular(fname, **kwargs):
    """Read a table from a .xlsx or .csv file

    Parameters
    ----------
    fname (string):

    Returns
    -------

    """
    file_format = fname.split(".")[-1]
    if file_format == "xlsx":
        df = pd.read_excel(fname, engine="openpyxl", **kwargs)
    elif file_format == "csv":
        df = pd.read_csv(fname, **kwargs)
    else:
        raise ValueError("Unsupported file type %s" % file_format)

    return df


def get_fpaths(fdir, idx_list, file_format="txt"):
    """Check the existence of files named by subject indices under a given directory

    Parameters
    ----------
    fdir (string): file directory for data
    idx_list (list): a list of subject indices
    file_format (string): file format, defaults to txt

    Returns
    -------
        Dataframe
    """
    fpaths = dict()
    for idx in idx_list:
        fname = "%s.%s" % (idx, file_format)
        fpath = os.path.join(fdir, fname)
        if os.path.exists(fpath):
            fpaths[idx] = fpath

    fpath_df = pd.DataFrame(data={"File path": fpaths.values()}, index=list(fpaths.keys()))

    return fpath_df


def split_functional_brain(matrix, connection_type="intra"):
    """

    Parameters
    ----------
    matrix
    connection_type

    Returns
    -------

    """
    n_ = matrix.shape[1] / 2
    if connection_type == "intra":
        left = matrix[0::2, 0::2]  # even indices of rows, left brain
        right = matrix[1::2, 1::2]  # odd indices of rows, right brain

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
    """Save two hemisphere nadarrays into a hdf5 file

    Parameters
    ----------
    out_dir (string):
    out_fname (string):
    data_left (ndarray):
    data_right (ndarray):

    Returns
    -------

    """
    f = h5py.File(os.path.join(out_dir, out_fname), "w")
    f.create_dataset("Left", data=data_left)
    f.create_dataset("Right", data=data_right)
    f.close()


def save_half_brain_mat(out_dir, out_fname, data_left, data_right):
    """Save two hemisphere nadarrays into a hdf5 file

    Parameters
    ----------
    out_dir (string):
    out_fname (string):
    data_left (ndarray):
    data_right (ndarray):

    Returns
    -------

    """
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
    """

    Parameters
    ----------
    data_dir
    atlas
    session
    run
    connection_type
    data_type
    dataset
    download: bool, optional, download the data if the data files not exist (default=True)

    Returns
    -------

    """
    if dataset == "HCP":
        if data_type == "functional":
            fname = "HCP_%s_%s_half_brain_%s_%s.hdf5" % (
                atlas,
                connection_type,
                session,
                run,
            )
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                if download:
                    os.makedirs(data_dir, exist_ok=True)
                    print("Downloading %s session %s data, it may take 1-2 mins" % (dataset, session))
                    download_url_to_file(
                        HCP_LINK[session],
                        fpath,
                    )
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
        fpath = os.path.join(
            data_dir,
            "%s_%s_%s_half_brain_%s.mat" % (dataset, atlas, connection_type, run),
        )
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


def fetch_weights(base_dir, group, lambda_, dataset, sessions, test_size="00", seed_=2023):
    """

    Args:
        base_dir:
        group: int, 0 or 1
        lambda_: str, "0_group_mix" or "0", "1", "2", "5", "8", "10"
        dataset: str, "HCP" or "GSP"
        sessions: list of session names, ["REST1_", "REST2_"] for HCP and ["_"] for GSP
        test_size: str, "00" or "02", optional, default="00"
        seed_: int, optional, default=2023

    Returns:
        a matrix of weights, shape (n_models, n_features)
    """

    sub_dir = os.path.join(base_dir, "lambda%s" % lambda_)
    if not os.path.exists(sub_dir):
        return None
    if lambda_ == "0_group_mix":
        lambda_ = 0
        group = "mix"
    weight = []
    num_repeat = 5
    halfs = [0, 1]
    for session_i in sessions:
        for half_i in halfs:
            for i_split in range(num_repeat):
                for seed in range(52):
                    model_file_name = "%s_L%s_test_size%s_%s%s_%s_group_%s_%s.pt" % (
                        dataset,
                        lambda_,
                        test_size,
                        session_i,
                        i_split,
                        half_i,
                        group,
                        seed_ - seed,
                    )
                    if os.path.exists(os.path.join(sub_dir, model_file_name)):
                        weight.append(get_coef(model_file_name, sub_dir).reshape((1, -1)))

    return np.concatenate(weight, axis=0)


def get_coef(file_name, file_dir):
    file_path = os.path.join(file_dir, file_name)
    model = torch.load(file_path)
    return model.theta


def save_results(res_dict, out_filename, output_dir, mix_group=False):
    """save a dictionary to a csv file

    Args:
        res_dict (_type_): _description_
        out_filename (str): _description_
        output_dir (str): _description_
        mix_group (bool, optional): _description_. Defaults to False.
    """
    res_df = pd.DataFrame.from_dict(res_dict)

    if mix_group:
        out_filename = out_filename + "_mix_group"
    out_file = os.path.join(output_dir, "%s.csv" % out_filename)
    res_df.to_csv(out_file, index=False)


# read: nib.load()
# write: nib.save()
# ********************************** file split ************************************* #
def file_split(file_path):
    filepath, tempfilename = os.path.split(file_path)
    filename, extension = os.path.splitext(tempfilename)
    return filepath, filename, extension


# ********************************** read h5 or pickle file as a df ************************************* #


def read_file(file):
    _, _, ext = file_split(file)
    if ext == ".h5":
        df = pd.read_hdf(file)
    elif ext == ".pkl":
        df = pd.read_pickle(file)
    else:
        raise ValueError("Unsupported file type %s, supported type: xxx.h5 or xxx.pkl..." % ext)

    return df


def select_data_subj(df, subj_id, df_column_name):
    data = df[df_column_name][df["subject"] == subj_id]
    return data  # series datatype with dim of (1,),for the ndarray contents, data = data[0]


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
    train_idx, hold_idx = train_test_split(range(n_), test_size=0.5, random_state=random_state)
    x = np.concatenate([data["Left"][train_idx], data["Right"][train_idx]], axis=0)
    y = np.ones(n_)
    y[int(n_ / 2) :] = -1

    return x, y


def select_data_multi_subj(df, subj_ids, df_column_name):
    subj_df = df["subject"].to_numpy()
    sub, index1, index2 = np.intersect1d(subj_df, subj_ids, assume_unique=False, return_indices=True)
    data = df.loc[index1, df_column_name]  # loc,select column name. iloc,select data as index
    return data.to_numpy()  # convert pd series to numpy ndarray


# ********************************** read MRI files ********************************* #
def read_surf(surface):
    surf = nib.load(surface)
    arr = surf.darrays
    coord = arr[0].data  # coordinate of vertices
    vert = arr[1].data  # index of vertices
    return coord, vert


def read_nii(T1w):
    T1 = nib.load(T1w)
    data = T1.get_fdata()  # 3-D ‘ray
    header = T1.header  # header
    return data, header


def read_shape_gii(shape_gii):
    gii = nib.load(shape_gii)
    data = gii.darrays[0].data
    return gii, data


# ********************************** Creat a .shape.gii file ********************************* #


def creat_shape_gii(shape_gii_base, array_write, savepath):
    gii, _ = read_shape_gii(shape_gii_base)
    gii.darrays[0].data = array_write
    nib.save(gii, savepath)


# Creat a .shape gii file with atlas, e.g.,Kong400,Schaefer400
def creat_fs_lr32k_atlas(shape_gii_base, array_write, atlas_file, hemi, savepath):
    # cdata = np.zeros((32492,))
    f_atlas = nib.load(atlas_file)
    f_data = f_atlas.get_fdata()  # 0 mean medial wall, label starts from 1
    f_data = f_data[0, :]

    if hemi == "L":
        cdata = f_data[0 : int(len(f_data) / 2)].copy()  # L
        print("cdata.shape: %d" % (len(cdata)))
        for i, data in enumerate(array_write):
            cdata[np.where(cdata == i + 1)] = data
    else:
        cdata = f_data[int(len(f_data) / 2) : int(len(f_data))].copy()  # R
        for i, data in enumerate(array_write):
            cdata[np.where(cdata == i + 1 + int(np.max(f_data) / 2))] = data

    creat_shape_gii(shape_gii_base, cdata, savepath)


# *********************************************** Corr with pvalue ************************************************ #
def Corr(x, y):
    r, p = pearsonr(x, y)
    rr = round(r, 3)
    pp = round(p, 3)
    return rr, pp


# ************************************************ Join plot ***************************************************** #
# x,y should not be the object data type, if so, convert it to np.double data type.
def jointplot_fitlinear(x, y):
    plt.figure(figsize=(20, 16))
    sns.jointplot(x=x, y=y, kind="reg", scatter_kws={"s": 8})


# sns.regplot(x='LL_LPAC_Math_Corr',y='ReadEng_AgeAdj',data=df, color='red', scatter_kws={"s": 8}),不显示分布信息


# *************************************** Extract top N in a array ************************************ #
# Extract top N element in an array and set others to 0.
def topN_array(arr, topN):
    idx_all = set(np.arange(len(arr)))
    idx_topN = arr.argsort()[::-1][:topN]
    idx_res = idx_all - set(idx_topN)
    # keep topN, set others to zero
    idx_res = np.array(list(idx_res))
    arr_topN = arr.copy()
    arr_topN[idx_res] = 0
    return arr_topN


# def main_creat_shape_gii():
#
#     tasks = ['LANGUAGE']
#     subjs = sio.loadmat(project_path + '/Subject_taskT.mat')
#     subjects = subjs['Subject_taskT']
#     subjects = np.reshape(subjects,(len(subjects),))
#     for creat shape gii files
#     shape_gii_base_L = project_path + '/Base_shape_gii/100206.L.thickness.32k_fs_LR.shape.gii'
#     for creat shape gii files
#     shape_gii_base_R = project_path + '/Base_shape_gii/100206.R.thickness.32k_fs_LR.shape.gii'
#
#     for subj_id in subjects:
#         df_column_name = 'task_t'
#         for task in tasks:
#             files = sorted(glob.glob(project_path + '/DataSort/Sort_Sub/' + task + '/*.h5'))
#
#             for file in files:
#
#                 Dirs = os.path.split(os.path.splitext(file)[0])
#                 kName = Dirs[1]
#                 df = rvp.read_orig_data(file, kName)
#                 ResultantFolder = (project_path + '/Result/Ridge_Vert_SubBased/Statistics/task_t_gii/'
#                                    + str(subj_id) +'/' + task)
#                 if not os.path.exists(ResultantFolder):
#                     os.makedirs(ResultantFolder)
#                 task_t = rvp.select_data_subj(df, subj_id, 'task_t').to_numpy()[0]
#                 task_t = np.reshape(task_t, (len(task_t),))
#                 vert_index = rvp.select_data_subj(df, subj_id, 'vert_index').to_numpy()[0]
#                 cdata = np.zeros((32492,))
#                 vert_index = np.reshape(vert_index, (len(vert_index),))
#                 cdata[vert_index - 1] = task_t # compared to matlab, the vert index should minus 1
#
#                 save_name = ResultantFolder + '/' + str(subj_id) + '_' + kName + '.shape.gii'
#                 if 'LW' in file:
#                     print('LEFT SEED')
#                     shape_gii_base= shape_gii_base_L
#                 elif 'RW' in file:
#                     print('RIGHT SEED')
#                     shape_gii_base= shape_gii_base_R
#                 else:
#                     print('error!')
#
#                 creat_shape_gii(shape_gii_base, cdata, save_name)
#                 print(kName)
#
#         print(str(subj_id))
