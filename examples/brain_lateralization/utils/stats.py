import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr


def corr(x, y):
    r, p = pearsonr(x, y)
    return round(r, 3), round(p, 3)


def Corr(x, y):
    return corr(x, y)


def jointplot_fitlinear(x, y):
    plt.figure(figsize=(20, 16))
    sns.jointplot(x=x, y=y, kind="reg", scatter_kws={"s": 8})


def top_n_array(arr, top_n):
    idx_all = set(np.arange(len(arr)))
    idx_top_n = arr.argsort()[::-1][:top_n]
    idx_res = idx_all - set(idx_top_n)
    idx_res = np.array(list(idx_res))
    arr_top_n = arr.copy()
    arr_top_n[idx_res] = 0
    return arr_top_n


def topN_array(arr, topN):
    return top_n_array(arr, topN)
