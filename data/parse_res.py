import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src import visualization as vis
from src.filter.ekf import Ekf
from src.smoother.ext.eks import Eks
from src.smoother.ext.ieks import Ieks
from src.smoother.ext.lm_ieks import LmIeks
from src.smoother.slr.ipls import SigmaPointIpls
from src.utils import setup_logger
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import CoordTurn
from src.smoother.slr.lm_ipls import SigmaPointLmIpls
from src.slr.sigma_points import SigmaPointSlr
from src.sigma_points import SphericalCubature
from src.cost import analytical_smoothing_cost, slr_smoothing_cost
from src.analytics import rmse, nees
from src.visualization import to_tikz, write_to_tikz_file
from data.lm_ieks_paper.coord_turn_example import simulate_data
from exp.coord_turn.common import plot_results, plot_cost


def main():
    smoothers = [
        # "ieks",
        # "lm-ieks",
        "ipls",
        "lm-ipls",
    ]
    rmse_stats = [(np.loadtxt(Path.cwd() / f"results/rmse/{label}.csv"), label.upper()) for label in smoothers]
    nees_stats = [(np.loadtxt(Path.cwd() / f"results/nees/{label}.csv"), label.upper()) for label in smoothers]
    plot_metric(rmse_stats, "RMSE")
    plot_metric(nees_stats, "NEES")
    # tikz_stats(Path.cwd() / "tmp_results/corrected", "LM-IEKS", rmse_stats)


def tikz_stats(dir_, name, stats):
    num_iter = stats[0][0].shape[1]
    iter_range = np.arange(1, num_iter + 1)
    stats = [(mc_stats(stat_), label) for stat_, label in stats]
    for (mean, err), label in stats:
        data = np.column_stack((iter_range, mean, err))
        np.savetxt(dir_ / name.lower() / f"{label}.csv", data)


def save_stats(res_dir, name, stats):
    for stat, label in stats:
        np.savetxt(res_dir / name.lower() / f"{label}.csv", stat)


def plot_metric(stats, title):
    num_iter = stats[0][0].shape[1]
    stats = [(mc_stats(stat_), label) for stat_, label in stats]
    fig, ax = plt.subplots()
    for (mean, err), label in stats:
        ax.errorbar(x=np.arange(1, num_iter + 1), y=mean, yerr=err, label=label)
    ax.set_title(title)
    ax.legend()
    plt.show()


def mc_stats(data):
    num_mc_samples = data.shape[0]
    return np.mean(data, 0), np.std(data, 0) / np.sqrt(num_mc_samples)


if __name__ == "__main__":
    main()
