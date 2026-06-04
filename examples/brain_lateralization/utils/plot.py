import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from utils.io_ import fetch_weights


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


# def plot_gsi(
#     data,
#     x="lambda",
#     y="GSI",
#     hue="train_gender",
#     fontsize=14,
#     outfile=None,
#     outfig_format=None,
# ):
#     fig = plt.figure()
#     plt.rcParams.update({"font.size": fontsize})
#     sns.boxplot(data=data, x=x, y=y, hue=hue, showmeans=True)
#     plt.legend(title="Target group")
#     plt.rcParams["text.usetex"] = True
#     plt.xlabel(r"$\lambda$")
#     plt.rcParams["text.usetex"] = False
#     plt.ylabel("Group Specificity Index (GSI)")
#     if outfile is not None:
#         savefig(fig, outfile, outfig_format)
#     plt.show()
#
#
# def plot_accuracy(
#     data,
#     x="Lambda",
#     y="Accuracy",
#     col="Target group",
#     hue="Test set",
#     style="Test set",
#     kind="line",
#     height=4,
#     fontsize=14,
#     outfile=None,
#     outfig_format=None,
# ):
#     fig = plt.figure()
#     plt.rcParams.update({"font.size": fontsize})
#     g = sns.relplot(
#         data=data,
#         x=x,
#         y=y,
#         col=col,
#         hue=hue,
#         style=style,
#         kind=kind,
#         errorbar=("sd", 1),
#         height=height,
#     )
#     (
#         g.map(plt.axhline, y=0.9, color=".7", dashes=(2, 1), zorder=0)
#         .set_axis_labels(r"$\lambda$", "Test Accuracy")
#         .set_titles("Target group: {col_name}")
#         .tight_layout(w_pad=0)
#     )
#     if outfile is not None:
#         savefig(fig, outfile, outfig_format)
#
#     plt.show()


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
            # n_weights = weights.shape[0]
            corr_matrix = np.corrcoef(control_weights, weights)[n_control_weights:, :n_control_weights]
            corrs["mean"].append(np.mean(corr_matrix))
            corrs["sd"].append(np.std(corr_matrix))

    plt.rcParams.update({"font.size": fontsize})
    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(4, 8.5)
    # .plot(x, y1)
    # .plot(x, y2)

    # plt.figure(figsize=(4, 4))
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
    # plt.savefig('figures/%s_corr.pdf' % dataset, format='pdf', bbox_inches='tight')
    # plt.savefig('figures/%s_corr.png' % dataset, format='png', bbox_inches='tight')
    plt.show()


def load_coef_plot_corr(dataset, group_label, fontsize=14):
    weights_dirs = dict()
    lambdas = [0.0, 1.0, 2.0, 5.0, 8.0, 10.0]
    group_dict = {0: "Male", 1: "Female"}
    for lambda_ in lambdas:
        weights_dirs[r"$\lambda=%s$" % (int(lambda_))] = "%s/%s_L%sG%s.mat" % (
            dataset,
            dataset,
            int(lambda_),
            group_label,
        )

    weights = dict()

    for key in weights_dirs:
        # print(key)
        weights[key] = loadmat(weights_dirs[key])["mean"][0][1:]

    weight_df = pd.DataFrame(weights)

    corr = weight_df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool), 1)

    plt.rcParams.update({"font.size": fontsize})
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    plt.rcParams["text.usetex"] = True
    # Draw the heatmap with the mask and correct aspect ratio
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
    # cbar_kws={"shrink": .5, "use_gridspec": False, "location": "top"})  #, annot_kws={"rotation": 45})
    plt.rcParams["text.usetex"] = False
    ax.set_xticklabels(weight_df.columns.to_list(), fontsize=fontsize + 2)
    ax.set_yticklabels(weight_df.columns.to_list(), rotation=45, ha="right", fontsize=fontsize + 2)
    # plt.savefig('corr.pdf', format='pdf', bbox_inches='tight')
    # plt.savefig('corr.png', format='png', bbox_inches='tight')
    plt.savefig(
        "figures/corr_annot_%s_%s.svg" % (dataset, group_dict[group_label]),
        format="svg",
        bbox_inches="tight",
    )
    # plt.savefig('figures/corr_annot_%s_%s.pdf' % (dataset, group_dict[sex_label]), format='pdf', bbox_inches='tight')
    # plt.savefig('figures/corr_annot_%s_%s.png' % (dataset, group_dict[sex_label]), format='png', bbox_inches='tight')
    plt.show()
