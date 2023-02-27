
import matplotlib.pyplot as plt
import numpy as np
from itertools import product


def plt_color(ax=None, color=None):
    if color is None:
        ax = ax or plt.gca()
        color = next(ax._get_lines.prop_cycler)["color"]
    return color




def plot_marginals(
    samples,
    ax_mat=None,
    kw_sc=None,
    kw_hist=None,
):
    (_,d) = samples.shape

    if ax_mat is None:
        _, ax_mat = plt.subplots(d, d)

    s = 0 if samples is None else len(samples)
    s = 20 / np.log10(10 + 10 * s)
    kw_sc = {
        **{"s": s, "alpha": 0.3, "marker": "."},
        **(kw_sc or {}),
    }
    kw_hist = {
        **{"density": True, "bins": "sqrt"},
        **(kw_hist or {}),
    }  # "edgecolor": "black", "histtype": "stepfilled",
    for i, j in product(range(d), repeat=2):
        ax: plt.Axes = ax_mat[d - 1 - j][i]
        k = 1
        if i != j:
            kw = {**kw_sc}
            kw["color"] = plt_color(ax, kw.get("color"))
            ax.scatter(samples[:, i], samples[:, j], **kw)
        else:
            kw = {**kw_hist}
            kw["color"] = plt_color(ax, kw.get("color"))
            ax.hist(samples[:, i], **kw)
    return ax_mat