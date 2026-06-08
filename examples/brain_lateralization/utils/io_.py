"""Compatibility imports for the brain lateralization utility modules.

Prefer importing from the focused modules directly:
``data_io``, ``half_brain``, ``results``, ``neuro_io``, and ``stats``.
"""

from .data_io import (
    file_split,
    get_fpaths,
    load_hdf5,
    load_result,
    load_txt,
    read_file,
    read_tabular,
    reformat_results,
    select_data_multi_subj,
    select_data_subj,
)
from .half_brain import (
    _pick_half_subs,
    GSP_LINK,
    HCP_LINK,
    load_half_brain,
    pick_half,
    save_half_brain,
    save_half_brain_mat,
    split_functional_brain,
)
from .neuro_io import (
    creat_fs_lr32k_atlas,
    creat_shape_gii,
    create_fs_lr32k_atlas,
    create_shape_gii,
    read_nii,
    read_shape_gii,
    read_surf,
)
from .results import fetch_weights, get_coef, save_results
from .stats import Corr, corr, jointplot_fitlinear, top_n_array, topN_array

__all__ = [
    "Corr",
    "GSP_LINK",
    "HCP_LINK",
    "_pick_half_subs",
    "corr",
    "creat_fs_lr32k_atlas",
    "creat_shape_gii",
    "create_fs_lr32k_atlas",
    "create_shape_gii",
    "fetch_weights",
    "file_split",
    "get_coef",
    "get_fpaths",
    "jointplot_fitlinear",
    "load_half_brain",
    "load_hdf5",
    "load_result",
    "load_txt",
    "pick_half",
    "read_file",
    "read_nii",
    "read_shape_gii",
    "read_surf",
    "read_tabular",
    "reformat_results",
    "save_half_brain",
    "save_half_brain_mat",
    "save_results",
    "select_data_multi_subj",
    "select_data_subj",
    "split_functional_brain",
    "topN_array",
    "top_n_array",
]
