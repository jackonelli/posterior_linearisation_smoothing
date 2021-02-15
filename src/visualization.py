"""Vizualisation"""
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from src.analytics import nees
import tikzplotlib


def plot_states(ax, x, label):
    D_x = x.shape[1]
    for d in range(D_x):
        ax.plot(x[:, d], label=f"x_{d}")


def cmp_states(seq_1, seq_2):
    D_x = seq_1.shape[1]
    _, ax = plt.subplots()
    for d in range(D_x):
        ax.plot(seq_1[:, d], "-", label=f"x_{d}")
        ax.plot(seq_2[:, d], "--", label=f"x_{d}")
    plt.show()


def plot_nees_comp(true_x, m_1, P_1, m_2, P_2):
    nees_1 = nees(true_x, m_1, P_1)
    nees_2 = nees(true_x, m_2, P_2)
    _, ax = plt.subplots()
    ax.plot(nees_1, "-b", label="kf")
    ax.plot(nees_2, "--g", label="slr")
    plt.show()


def plot_1d_est(true_x, meas, means_and_covs, sigma_level=3, skip_cov=1, ax=None):
    time_steps = np.arange(true_x.shape[0])
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(time_steps, true_x, ".k", label="true")

    if meas is not None:
        ax.plot(time_steps, meas, ".r", label="meas")

    for m, P, label in means_and_covs:
        plot_1d_mean_and_cov(ax, m, P, sigma_level, label, "b", skip_cov)

    ax.set_title("Estimates")
    ax.set_xlabel("$pos_x$")
    ax.set_ylabel("$pos_y$")
    ax.legend()


def plot_1d_mean_and_cov(ax, means, covs, sigma_level, label, color, skip_cov):
    time_steps = np.arange(means.shape[0])
    stds = np.sqrt(covs)
    handle = ax.errorbar(x=time_steps, y=means, yerr=sigma_level * stds)
    handle.set_label(r"{}, ${} \sigma$".format(label, sigma_level))
    return handle


def plot_2d_est(true_x, meas, means_and_covs, sigma_level=3, skip_cov=1, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(true_x[:, 0], true_x[:, 1], ".k", label="true")

    if meas is not None:
        ax.plot(meas[:, 0], meas[:, 1], ".r", label="meas")

    for m, P, label in means_and_covs:
        plot_2d_mean_and_cov(ax, m[:, :2], P[:, :2, :2], sigma_level, label, skip_cov)

    ax.set_title("Estimates")
    ax.set_xlabel("$pos_x$")
    ax.set_ylabel("$pos_y$")
    ax.legend()


def plot_nees_and_2d_est(true_x, meas, means_and_covs, sigma_level=3, skip_cov=1):
    K, D_x = true_x.shape
    _, (ax_1, ax_2) = plt.subplots(1, 2)
    ax_1.plot([0, K], [D_x, D_x], "--k", label="ref")
    ax_2.plot(true_x[:, 0], true_x[:, 1], ".k", label="true")

    if meas is not None:
        ax_2.plot(meas[:, 0], meas[:, 1], ".r", label="meas")

    for m, P, label in means_and_covs:
        traj_handle = plot_2d_mean_and_cov(ax_2, m[:, :2], P[:, :2, :2], sigma_level, label, skip_cov)
        filter_nees = nees(true_x, m, P)
        (nees_handle,) = ax_1.plot(filter_nees, "-b", label=label)
        nees_handle.set_color(traj_handle.get_color())

    ax_1.set_title("NEES")
    ax_1.set_xlabel("k")
    ax_1.set_ylabel(r"$\epsilon_{x, k}$")
    ax_1.legend()

    ax_2.set_title("Estimates")
    ax_2.set_xlabel("$pos_x$")
    ax_2.set_ylabel("$pos_y$")
    ax_2.legend()
    plt.show()


def plot_2d_mean_and_cov(ax, means, covs, sigma_level, label, skip_cov):
    fmt = "-"
    (handle,) = ax.plot(means[:, 0], means[:, 1], fmt, label=label)
    color = handle.get_color()
    for k in np.arange(0, len(means), skip_cov):
        last_handle = plot_sigma_level(ax, means[k, :], covs[k, :, :], sigma_level, "", color)
    last_handle.set_label(r"${} \sigma$".format(sigma_level))
    return handle


def plot_sigma_level(ax, means, covs, level, label, color, resolution=50):
    fmt = "--"
    ellips = ellips_points(means, covs, level, resolution)
    (handle,) = ax.plot(ellips[:, 0], ellips[:, 1], fmt)
    handle.set_color(color)
    return handle


def ellips_points(center, transf, scale, resolution):
    """Transform the circle to the sought ellipse"""
    angles = np.linspace(0, 2 * np.pi, resolution)
    curve_parameter = np.row_stack((np.cos(angles), np.sin(angles)))

    level_sigma_offsets = scale * sqrtm(transf) @ curve_parameter

    return center + level_sigma_offsets.T


def to_tikz(fig):
    tikzplotlib.clean_figure(fig)
    output = tikzplotlib.get_tikz_code(
        figure=fig, filepath=None, axis_width=None, axis_height=None, textsize=10.0, table_row_sep="\n"
    )
    return output


def write_to_tikz_file(tikz_string, tikz_dir, name):
    file_name = tikz_dir / (name + ".tikz")
    with open(file_name, "w") as tikz_file:
        tikz_file.write(tikz_string)
