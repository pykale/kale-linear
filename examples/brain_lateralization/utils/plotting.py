import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat

from .results import fetch_weights


def savefig(fig, outfile, outfig_format):
    if isinstance(outfig_format, list):
        for fmt in outfig_format:
            fig.savefig("%s.%s" % (outfile, fmt), format=fmt, bbox_inches="tight")
    else:
        fig.savefig(
            "%s.%s" % (outfile, outfig_format),
            format=outfig_format,
            bbox_inches="tight",
        )


def load_weight_plot_corr(dataset, base_dir, sessions, seed_start, fontsize=14):
    control_weights = fetch_weights(base_dir, "mix", "0_group_mix", dataset, sessions=sessions, seed_=seed_start)
    n_control_weights = control_weights.shape[0]

    lambdas = [0.0, 1.0, 2.0, 5.0, 8.0, 10.0]
    corrs = {"mean": [], "sd": []}

    for lambda_ in lambdas:
        for group in [0, 1]:
            weights = fetch_weights(
                base_dir,
                group,
                int(lambda_),
                dataset,
                sessions=sessions,
                seed_=seed_start,
            )
            corr_matrix = np.corrcoef(control_weights, weights)[n_control_weights:, :n_control_weights]
            corrs["mean"].append(np.mean(corr_matrix))
            corrs["sd"].append(np.std(corr_matrix))

    plt.rcParams.update({"font.size": fontsize})
    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(4, 8.5)

    ax[0].plot(lambdas, corrs["mean"][::2], "-", c="steelblue", label="Male-specific")
    ax[0].fill_between(
        lambdas,
        np.asarray(corrs["mean"][::2]) - np.asarray(corrs["sd"][::2]),
        np.asarray(corrs["mean"][::2]) + np.asarray(corrs["sd"][::2]),
        color="steelblue",
        alpha=0.2,
    )
    ax[0].set_ylabel("Correlation")
    ax[0].legend(loc="upper right")

    ax[1].plot(lambdas, corrs["mean"][1::2], "--", color="firebrick", label="Female-specific")
    ax[1].fill_between(
        lambdas,
        np.asarray(corrs["mean"][1::2]) - np.asarray(corrs["sd"][1::2]),
        np.asarray(corrs["mean"][1::2]) + np.asarray(corrs["sd"][1::2]),
        color="firebrick",
        alpha=0.2,
    )

    ax[1].set_ylabel("Correlation")
    ax[1].legend(loc="upper right")
    plt.rcParams["text.usetex"] = True
    plt.xlabel(r"$\lambda$", fontsize=fontsize)
    plt.rcParams["text.usetex"] = False

    plt.savefig("figures/%s_corr.svg" % dataset, format="svg", bbox_inches="tight")
    plt.show()


def load_coef_plot_corr(dataset, group_label, fontsize=14):
    weights_dirs = {}
    lambdas = [0.0, 1.0, 2.0, 5.0, 8.0, 10.0]
    group_dict = {0: "Male", 1: "Female"}
    for lambda_ in lambdas:
        weights_dirs[r"$\lambda=%s$" % (int(lambda_))] = "%s/%s_L%sG%s.mat" % (
            dataset,
            dataset,
            int(lambda_),
            group_label,
        )

    weights = {}
    for key in weights_dirs:
        weights[key] = loadmat(weights_dirs[key])["mean"][0][1:]

    weight_df = pd.DataFrame(weights)
    corr = weight_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), 1)

    plt.rcParams.update({"font.size": fontsize})
    _, ax = plt.subplots(figsize=(11, 9))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    plt.rcParams["text.usetex"] = True
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=0,
        vmax=1,
        center=0.5,
        annot=True,
        annot_kws={"fontsize": "xx-large"},
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    plt.rcParams["text.usetex"] = False
    ax.set_xticklabels(weight_df.columns.to_list(), fontsize=fontsize + 2)
    ax.set_yticklabels(weight_df.columns.to_list(), rotation=45, ha="right", fontsize=fontsize + 2)
    plt.savefig(
        "figures/corr_annot_%s_%s.svg" % (dataset, group_dict[group_label]),
        format="svg",
        bbox_inches="tight",
    )
    plt.show()
