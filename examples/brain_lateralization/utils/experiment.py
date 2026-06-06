import copy
import os
import sys

# import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score  # , roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.hub import download_url_to_file

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from kalelinear.estimator import GSDA  # , GSLRTorch

from .data_io import read_tabular
from .half_brain import load_half_brain, pick_half

BASE_RESULT_DICT: dict = {
    "pred_loss": [],
    "hsic_loss": [],
    "lambda": [],
    "train_session": [],
    "split": [],
    "fold": [],
    "target_group": [],
    "time_used": [],
}

LABEL_FILE_LINK = {
    "HCP": "https://zenodo.org/records/10050233/files/HCP_half_brain.csv",
    "GSP": "https://zenodo.org/records/10050234/files/gsp_half_brain.csv",
}

GSDA_INIT_ARGS = ["lr", "max_iter", "l2_hparam", "lambda_", "optimizer"]

GROUP_DICT = {0: "Male", 1: "Female", "mix": "Mix"}


def get_gsda_init_kws(lambda_, l2_hparam, extra_kws):
    init_kws = {"lambda_": lambda_, "l2_hparam": l2_hparam}
    for arg in GSDA_INIT_ARGS:
        if arg in extra_kws:
            init_kws[arg] = extra_kws[arg]

    return init_kws


def get_target_groups(mix_group):
    if mix_group:
        return ["mix"]

    return [0, 1]


def get_fit_kws(y, groups, target_idx=None):
    return {
        "y": y,
        "groups": groups,
        "target_idx": target_idx,
    }


def run_experiment(cfg, lambda_):
    atlas = cfg.DATASET.ATLAS
    download = cfg.DATASET.DOWNLOAD
    connection_type = cfg.DATASET.CONNECTION
    data_dir = cfg.DATASET.ROOT
    dataset = cfg.DATASET.DATASET.upper()
    # lambda_list = cfg.SOLVER.LAMBDA_
    run_ = cfg.DATASET.RUN
    random_state = cfg.SOLVER.SEED
    test_size = cfg.DATASET.TEST_RATIO

    l2_hparam = cfg.SOLVER.L2_HPARAM
    lr = cfg.SOLVER.LR
    optimizer = cfg.SOLVER.OPTIMIZER
    num_repeat = cfg.DATASET.NUM_REPEAT
    max_iter = cfg.SOLVER.MAX_ITER
    mix_group = cfg.DATASET.MIX_GROUP
    rest1_only = cfg.DATASET.REST1_ONLY

    if mix_group:
        # lambda_list = [0.0]
        lambda_ = 0.0

    label_file = "%s_%s_half_brain.csv" % (cfg.DATASET.DATASET, atlas)
    label_fpath = os.path.join(data_dir, label_file)
    if not os.path.exists(label_fpath):
        if download:
            os.makedirs(data_dir, exist_ok=True)
            print("Downloading label file for %s dataset. \n" % dataset)
            download_url_to_file(LABEL_FILE_LINK[dataset], label_fpath)
        else:
            raise ValueError("File %s does not exist" % label_file)
    labels = read_tabular(label_fpath, index_col="ID")
    group_label = labels["gender"].values

    # for lambda_ in lambda_list:
    if mix_group:
        out_folder = "lambda0_group_mix"
    else:
        out_folder = "lambda%s" % int(lambda_)
    out_dir = os.path.join(cfg.OUTPUT.ROOT, out_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    kwargs = {
        "groups": group_label,
        "lambda_": lambda_,
        "l2_hparam": l2_hparam,
        "lr": lr,
        "mix_group": mix_group,
        "out_dir": out_dir,
        "num_repeat": num_repeat,
        "test_size": test_size,
        "random_state": random_state,
        "rest1_only": rest1_only,
        "optimizer": optimizer,
        "max_iter": max_iter,
    }

    if dataset == "HCP":
        data = dict()
        sessions = ["REST1", "REST2"]
        for session in sessions:
            data[session] = load_half_brain(data_dir, atlas, session, run_, connection_type, download=download)
        kwargs = {**{"data": data}, **kwargs}
        print("Training model with lambda = %s on %s dataset. \n" % (lambda_, dataset))
        if test_size == 0:
            results = run_no_sub_hold_hcp(**kwargs)
        elif 0 < test_size < 1:
            results = run_sub_hold_hcp(**kwargs)
        else:
            raise ValueError("Invalid test_size %s" % test_size)

    elif dataset == "GSP":
        data = load_half_brain(
            data_dir,
            atlas,
            session=None,
            run=run_,
            connection_type=connection_type,
            dataset=dataset,
            download=download,
        )
        kwargs = {**{"data": data}, **kwargs}
        print("Training model with lambda = %s on %s dataset. \n" % (lambda_, dataset))
        if 0 < test_size < 1:
            results = run_sub_hold_gsp(**kwargs)
        elif test_size == 0:
            results = run_no_sub_hold_gsp(**kwargs)
        else:
            raise ValueError("Invalid test_size %s." % test_size)
    else:
        raise ValueError("Invalid dataset %s." % dataset)

    res_df = pd.DataFrame.from_dict(results)
    out_filename = "results_%s_L%s_test_size0%s_%s_%s" % (
        dataset,
        int(lambda_),
        str(int(test_size * 10)),
        run_,
        random_state,
    )

    if mix_group:
        out_filename = out_filename + "_group_mix"
    out_file = os.path.join(out_dir, "%s.csv" % out_filename)
    print("Saving classification results to %s. \n" % out_file)
    res_df.to_csv(out_file, index=False)

    # return res_df, out_file


def train_model(
    x_train,
    init_kws,
    fit_kws,
    out_dir,
    model_filename,
):
    model_path = os.path.join(out_dir, "%s.pt" % model_filename)

    if os.path.exists(model_path):
        model = torch.load(model_path)
    else:
        model = GSDA(**init_kws)
        model.fit(x_train, **fit_kws)
        torch.save(model, model_path)
        print("Saving trained model to %s. \n" % model_path)

    return model


def save_loop_results(
    model,
    res_dict,
    xy_test,
    target_group,
    train_session=None,
    lambda_=0.0,
    i_split=0,
    train_fold=0,
):
    for acc_key in xy_test:
        test_x, test_y = xy_test[acc_key]
        y_pred_ = model.predict(test_x)
        acc_ = accuracy_score(test_y, y_pred_)
        res_dict[acc_key].append(acc_)

    res_dict["pred_loss"].append(model.losses["pred"][-1])
    if "hsic" in model.losses:
        res_dict["hsic_loss"].append(model.losses["hsic"][-1])
    else:
        res_dict["hsic_loss"].append(model.losses["code"][-1])

    # res['n_iter'].append(n_iter)
    # n_iter += 1
    res_dict["target_group"].append(GROUP_DICT[target_group])
    if train_session is not None:
        res_dict["train_session"].append(train_session)
    else:
        res_dict["train_session"].append(0)
    res_dict["split"].append(i_split)
    res_dict["fold"].append(train_fold)

    res_dict["lambda"].append(lambda_)
    # res['test_size'].append(test_size)
    res_dict["time_used"].append(model.losses["time"][-1])

    return res_dict


def split_by_group(groups, train_subject, test_subject, target_group=0):
    """Split the data by group.

    Args:
        groups (np.ndarray): Group labels.
        train_subject (np.ndarray): Training subjects indices.
        test_subject (np.ndarray): Testing subjects indices.
        target_group (int, optional): Target group. Defaults to 0.
    """
    train_sub_tgt_idx = np.where(groups[train_subject] == target_group)[0]
    train_sub_nt_idx = np.where(groups[train_subject] == 1 - target_group)[0]
    test_sub_tgt_idx = np.where(groups[test_subject] == target_group)[0]
    test_sub_nt_idx = np.where(groups[test_subject] == 1 - target_group)[0]

    return train_sub_tgt_idx, train_sub_nt_idx, test_sub_tgt_idx, test_sub_nt_idx


def run_no_sub_hold_hcp(
    data,
    groups,
    lambda_,
    l2_hparam,
    mix_group,
    out_dir,
    num_repeat,
    random_state,
    **kwargs,
):
    # main loop
    res = {
        **{
            "acc_tgt_train_session": [],
            "acc_tgt_test_session": [],
            "acc_nt_train_session": [],
            "acc_nt_test_session": [],
        },
        **copy.deepcopy(BASE_RESULT_DICT),
    }

    for train_session, test_session in [("REST1", "REST2"), ("REST2", "REST1")]:
        if train_session == "REST2" and kwargs["rest1_only"]:
            continue
        for i_split in range(num_repeat):
            x_all = dict()
            y_all = dict()
            for session in [train_session, test_session]:
                x, y, x1, y1 = pick_half(data[session], random_state=random_state * (i_split + 1))
                # x, y = _pick_half_subs(data[run_])

                x_all[session] = [x, x1]
                y_all[session] = [y, y1]

            for i_fold in [0, 1]:
                x_train_fold = x_all[train_session][i_fold]
                y_train_fold = y_all[train_session][i_fold]

                # scaler = StandardScaler()
                # scaler.fit(x_train_fold)
                for tgt_group in get_target_groups(mix_group):
                    group_idx = 0 if mix_group else tgt_group
                    tgt_idx = np.where(groups == group_idx)[0]
                    nt_idx = np.where(groups == 1 - group_idx)[0]

                    xy_test = {
                        "acc_tgt_train_session": [
                            x_all[train_session][1 - i_fold][tgt_idx],
                            y_all[train_session][1 - i_fold][tgt_idx],
                        ],
                        "acc_tgt_test_session": [
                            np.concatenate([x_all[test_session][i][tgt_idx] for i in range(2)]),
                            np.concatenate([y_all[test_session][i][tgt_idx] for i in range(2)]),
                        ],
                        "acc_nt_train_session": [
                            x_all[train_session][1 - i_fold][nt_idx],
                            y_all[train_session][1 - i_fold][nt_idx],
                        ],
                        "acc_nt_test_session": [
                            np.concatenate([x_all[test_session][i][nt_idx] for i in range(2)]),
                            np.concatenate([y_all[test_session][i][nt_idx] for i in range(2)]),
                        ],
                    }

                    fit_kws = get_fit_kws(y_train_fold[tgt_idx], groups, tgt_idx)
                    model_filename = "HCP_L%s_test_size00_%s_%s_%s_group_%s_%s" % (
                        int(lambda_),
                        train_session,
                        i_split,
                        i_fold,
                        tgt_group,
                        random_state,
                    )
                    if mix_group:
                        model_filename = model_filename + "_group_mix"
                        fit_kws = get_fit_kws(y_train_fold, groups)

                    init_kws = get_gsda_init_kws(lambda_, l2_hparam, kwargs)

                    model = train_model(
                        x_train_fold,
                        init_kws,
                        fit_kws,
                        out_dir,
                        model_filename,
                    )
                    res = save_loop_results(
                        model,
                        res,
                        xy_test,
                        tgt_group,
                        train_session,
                        lambda_,
                        i_split,
                        i_fold,
                    )

    return res


def run_sub_hold_hcp(
    data,
    groups,
    lambda_,
    l2_hparam,
    mix_group,
    out_dir,
    num_repeat,
    test_size,
    random_state,
    **kwargs,
):
    res = {
        **{
            "acc_tgt_train_session": [],
            "acc_tgt_test_session": [],
            "acc_nt_train_session": [],
            "acc_nt_test_session": [],
            "acc_tgt_test_sub": [],
            "acc_nt_test_sub": [],
        },
        **copy.deepcopy(BASE_RESULT_DICT),
    }
    x_all = dict()
    y_all = dict()

    for session in ["REST1", "REST2"]:
        x, y, x1, y1 = pick_half(data[session], random_state=random_state)

        x_all[session] = [x, x1]
        y_all[session] = [y, y1]

    for train_session, test_session in [("REST1", "REST2"), ("REST2", "REST1")]:
        if train_session == "REST2" and kwargs["rest1_only"]:
            continue
        sss = StratifiedShuffleSplit(n_splits=num_repeat, test_size=test_size, random_state=random_state)

        for i_split, (train_sub, test_sub) in enumerate(sss.split(x, groups)):
            for train_fold in [0, 1]:
                x_train_fold = x_all[train_session][train_fold]
                y_train_fold = y_all[train_session][train_fold]

                for tgt_group in get_target_groups(mix_group):
                    group_idx = 0 if mix_group else tgt_group
                    _idx = split_by_group(groups, train_sub, test_sub, group_idx)
                    (
                        train_sub_tgt_idx,
                        train_sub_nt_idx,
                        test_sub_tgt_idx,
                        test_sub_nt_idx,
                    ) = _idx

                    xy_test = {
                        "acc_tgt_train_session": [
                            x_all[train_session][1 - train_fold][train_sub][train_sub_tgt_idx],
                            y_all[train_session][1 - train_fold][train_sub][train_sub_tgt_idx],
                        ],
                        "acc_tgt_test_session": [
                            np.concatenate([x_all[test_session][i][train_sub][train_sub_tgt_idx] for i in range(2)]),
                            np.concatenate([y_all[test_session][i][train_sub][train_sub_tgt_idx] for i in range(2)]),
                        ],
                        "acc_nt_train_session": [
                            x_all[train_session][1 - train_fold][train_sub][train_sub_nt_idx],
                            y_all[train_session][1 - train_fold][train_sub][train_sub_nt_idx],
                        ],
                        "acc_nt_test_session": [
                            np.concatenate([x_all[test_session][i][train_sub][train_sub_nt_idx] for i in range(2)]),
                            np.concatenate([y_all[test_session][i][train_sub][train_sub_nt_idx] for i in range(2)]),
                        ],
                        "acc_tgt_test_sub": [[], []],
                        "acc_nt_test_sub": [[], []],
                    }

                    for _s in [train_session, test_session]:
                        for _fold in (0, 1):
                            xy_test["acc_tgt_test_sub"][0].append(x_all[_s][_fold][test_sub][test_sub_tgt_idx])
                            xy_test["acc_tgt_test_sub"][1].append(y_all[_s][_fold][test_sub][test_sub_tgt_idx])

                            xy_test["acc_nt_test_sub"][0].append(x_all[_s][_fold][test_sub][test_sub_nt_idx])
                            xy_test["acc_nt_test_sub"][1].append(y_all[_s][_fold][test_sub][test_sub_nt_idx])

                    for _key in ["acc_tgt_test_sub", "acc_nt_test_sub"]:
                        for _idx in range(2):
                            xy_test[_key][_idx] = np.concatenate(xy_test[_key][_idx])

                    fit_kws = get_fit_kws(
                        y_train_fold[train_sub][train_sub_tgt_idx], groups[train_sub], train_sub_tgt_idx
                    )
                    model_filename = "HCP_L%s_test_size0%s_%s_%s_%s_group_%s_%s" % (
                        int(lambda_),
                        str(int(test_size * 10)),
                        train_session,
                        i_split,
                        train_fold,
                        tgt_group,
                        random_state,
                    )
                    x_train = x_train_fold[train_sub]

                    if mix_group:
                        model_filename = model_filename + "_group_mix"
                        fit_kws = get_fit_kws(y_train_fold[train_sub], groups[train_sub])

                    init_kws = get_gsda_init_kws(lambda_, l2_hparam, kwargs)

                    model = train_model(
                        x_train,
                        init_kws,
                        fit_kws,
                        out_dir,
                        model_filename,
                    )
                    res = save_loop_results(
                        model,
                        res,
                        xy_test,
                        tgt_group,
                        train_session,
                        lambda_,
                        i_split,
                        train_fold,
                    )

    return res


def run_sub_hold_gsp(
    data,
    groups,
    lambda_,
    l2_hparam,
    mix_group,
    out_dir,
    num_repeat,
    test_size,
    random_state,
    **kwargs,
):
    res = {
        **{"acc_tgt": [], "acc_nt": [], "acc_tgt_test_sub": [], "acc_nt_test_sub": []},
        **copy.deepcopy(BASE_RESULT_DICT),
    }

    x, y, x1, y1 = pick_half(data, random_state=random_state)

    x_all = [x, x1]
    y_all = [y, y1]

    sss = StratifiedShuffleSplit(n_splits=num_repeat, test_size=test_size, random_state=random_state)
    for i_split, (train_sub, test_sub) in enumerate(sss.split(x, groups)):
        for train_fold in [0, 1]:
            x_train_fold = x_all[train_fold]
            y_train_fold = y_all[train_fold]
            x_test_fold = x_all[1 - train_fold]
            y_test_fold = y_all[1 - train_fold]

            for tgt_group in get_target_groups(mix_group):
                group_idx = 0 if mix_group else tgt_group
                _idx = split_by_group(groups, train_sub, test_sub, group_idx)
                (
                    train_sub_tgt_idx,
                    train_sub_nt_idx,
                    test_sub_tgt_idx,
                    test_sub_nt_idx,
                ) = _idx

                x_train_fold_test_tgt = x_test_fold[train_sub][train_sub_tgt_idx]
                y_train_fold_test_tgt = y_test_fold[train_sub][train_sub_tgt_idx]
                x_train_fold_test_nt = x_test_fold[train_sub][train_sub_nt_idx]
                y_train_fold_test_nt = y_test_fold[train_sub][train_sub_nt_idx]

                fit_kws = get_fit_kws(y_train_fold[train_sub_tgt_idx], groups, train_sub_tgt_idx)
                model_filename = "GSP_L%s_test_size0%s_%s_%s_group_%s_%s" % (
                    int(lambda_),
                    str(int(test_size * 10)),
                    i_split,
                    train_fold,
                    tgt_group,
                    random_state,
                )
                if 0 < test_size < 1:
                    x_train = x_train_fold[train_sub]
                    xy_test = {
                        "acc_tgt": [x_train_fold_test_tgt, y_train_fold_test_tgt],
                        "acc_nt": [x_train_fold_test_nt, y_train_fold_test_nt],
                        "acc_tgt_test_sub": [
                            np.concatenate(
                                (
                                    x_train_fold[test_sub][test_sub_tgt_idx],
                                    x_test_fold[test_sub][test_sub_tgt_idx],
                                )
                            ),
                            np.concatenate(
                                (
                                    y_train_fold[test_sub][test_sub_tgt_idx],
                                    y_test_fold[test_sub][test_sub_tgt_idx],
                                )
                            ),
                        ],
                        "acc_nt_test_sub": [
                            np.concatenate(
                                (
                                    x_train_fold[test_sub][test_sub_nt_idx],
                                    x_test_fold[test_sub][test_sub_nt_idx],
                                )
                            ),
                            np.concatenate(
                                (
                                    y_train_fold[test_sub][test_sub_nt_idx],
                                    y_test_fold[test_sub][test_sub_nt_idx],
                                )
                            ),
                        ],
                    }
                    model_filename = model_filename + "_test_sub_0%s" % str(int(test_size * 10))
                    fit_kws = get_fit_kws(
                        y_train_fold[train_sub][train_sub_tgt_idx], groups[train_sub], train_sub_tgt_idx
                    )
                else:
                    x_train = x_train_fold
                    xy_test = {
                        "acc_tgt": [x_train_fold_test_tgt, y_train_fold_test_tgt],
                        "acc_nt": [x_train_fold_test_nt, y_train_fold_test_nt],
                    }

                if mix_group:
                    model_filename = model_filename + "_group_mix"
                    if 0 < test_size < 1:
                        fit_kws = get_fit_kws(y_train_fold[train_sub], groups[train_sub])
                    else:
                        fit_kws = get_fit_kws(y_train_fold, groups)
                init_kws = get_gsda_init_kws(lambda_, l2_hparam, kwargs)

                model = train_model(
                    x_train,
                    init_kws,
                    fit_kws,
                    out_dir,
                    model_filename,
                )
                res = save_loop_results(
                    model,
                    res,
                    xy_test,
                    tgt_group,
                    lambda_=lambda_,
                    i_split=i_split,
                    train_fold=train_fold,
                )

    return res


def run_no_sub_hold_gsp(
    data,
    groups,
    lambda_,
    l2_hparam,
    mix_group,
    out_dir,
    num_repeat,
    random_state,
    **kwargs,
):
    res = {
        **{"acc_tgt": [], "acc_nt": []},
        **copy.deepcopy(BASE_RESULT_DICT),
    }
    for i_split in range(num_repeat):
        x, y, x1, y1 = pick_half(data, random_state=random_state * (i_split + 1))
        x_all = [x, x1]
        y_all = [y, y1]

        for i_fold in [0, 1]:
            x_train = x_all[i_fold]
            y_train = y_all[i_fold]

            # scaler = StandardScaler()
            # scaler.fit(x_train)
            for tgt_group in get_target_groups(mix_group):
                group_idx = 0 if mix_group else tgt_group
                tgt_idx = np.where(groups == group_idx)[0]
                nt_idx = np.where(groups == 1 - group_idx)[0]

                xy_test = {
                    "acc_tgt": [x_all[1 - i_fold][tgt_idx], y_all[1 - i_fold][tgt_idx]],
                    "acc_nt": [x_all[1 - i_fold][nt_idx], y_all[1 - i_fold][nt_idx]],
                }

                fit_kws = get_fit_kws(y_train[tgt_idx], groups, tgt_idx)
                model_filename = "GSP_L%s_test_size00_%s_%s_group_%s_%s" % (
                    int(lambda_),
                    i_split,
                    i_fold,
                    tgt_group,
                    random_state,
                )

                if mix_group:
                    model_filename = model_filename + "_group_mix"
                    fit_kws = get_fit_kws(y_train, groups)
                init_kws = get_gsda_init_kws(lambda_, l2_hparam, kwargs)

                model = train_model(
                    x_train,
                    init_kws,
                    fit_kws,
                    out_dir,
                    model_filename,
                )
                res = save_loop_results(
                    model,
                    res,
                    xy_test,
                    tgt_group,
                    lambda_=lambda_,
                    i_split=i_split,
                    train_fold=i_fold,
                )

    return res
