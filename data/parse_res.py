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
from src.analytics import rmse, nees, mc_stats
from src.visualization import to_tikz, write_to_tikz_file, plot_scalar_metric_err_bar
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
    plot_scalar_metric_err_bar(rmse_stats, "RMSE")
    plot_scalar_metric_err_bar(nees_stats, "NEES")
    # tikz_stats(Path.cwd() / "tmp_results/corrected", "LM-IEKS", rmse_stats)


if __name__ == "__main__":
    main()
