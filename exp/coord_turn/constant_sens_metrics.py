"""Example:
Reproducing the experiment in the paper:
"Levenberg-Marquardt and line-search extended Kalman smoother".

The exact realisation of the experiment in the paper is found in
`exp/lm_ieks_paper.py`
"""

import argparse
import logging
from functools import partial
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src import visualization as vis
from src.filter.ekf import Ekf
from src.smoother.ext.eks import Eks
from src.smoother.ext.ieks import Ieks
from src.smoother.ext.lm_ieks import LmIeks
from src.smoother.slr.ipls import SigmaPointIpls
from src.utils import setup_logger, tikz_err_bar_tab_format
from src.models.range_bearing import MultiSensorBearings, MultiSensorRange
from src.models.coord_turn import CoordTurn
from src.smoother.slr.lm_ipls import SigmaPointLmIpls
from src.slr.sigma_points import SigmaPointSlr
from src.sigma_points import SphericalCubature
from src.cost import analytical_smoothing_cost, slr_smoothing_cost_pre_comp, slr_noop_cost
from src.analytics import rmse, nees
from src.visualization import to_tikz, write_to_tikz_file
from data.lm_ieks_paper.coord_turn_example import simulate_data
from exp.coord_turn.common import MeasType, run_smoothing, mc_stats, calc_iter_metrics


def main():
    log = logging.getLogger(__name__)
    args = parse_args()
    experiment_name = "ct_constant_sens"
    setup_logger(f"logs/{experiment_name}.log", logging.INFO)
    log.info(f"Running experiment: {experiment_name}")
    np.random.seed(0)
    dt = 0.01
    qc = 0.01
    qw = 10
    Q = np.array(
        [
            [qc * dt ** 3 / 3, 0, qc * dt ** 2 / 2, 0, 0],
            [0, qc * dt ** 3 / 3, 0, qc * dt ** 2 / 2, 0],
            [qc * dt ** 2 / 2, 0, qc * dt, 0, 0],
            [0, qc * dt ** 2 / 2, 0, qc * dt, 0],
            [0, 0, 0, 0, dt * qw],
        ]
    )
    motion_model = CoordTurn(dt, Q)

    sens_pos_1 = np.array([-1.5, 0.5])
    sens_pos_2 = np.array([1, 1])
    sensors = np.row_stack((sens_pos_1, sens_pos_2))
    std = 0.5
    R = std ** 2 * np.eye(2)
    meas_model = MultiSensorBearings(sensors, R)

    prior_mean = np.array([0, 0, 1, 0, 0])
    prior_cov = np.diag([0.1, 0.1, 1, 1, 1])

    num_iter = args.num_iter
    if args.meas_type == MeasType.Range:
        meas_model = MultiSensorRange(sensors, R)
        log.info("Using measurement model: MultiSensorRange")
    elif args.meas_type == MeasType.Bearings:
        meas_model = MultiSensorBearings(sensors, R)
        log.info("Using measurement model: MultiSensorBearings")

    states, measurements = simulate_data(motion_model, meas_model, prior_mean[:-1], time_steps=500)

    num_mc_samples = args.num_mc_samples

    rmses_gn_ieks = np.zeros((num_mc_samples, num_iter))
    rmses_lm_ieks = np.zeros((num_mc_samples, num_iter))
    rmses_gn_ipls = np.zeros((num_mc_samples, num_iter))
    rmses_lm_ipls = np.zeros((num_mc_samples, num_iter))

    neeses_gn_ieks = np.zeros((num_mc_samples, num_iter))
    neeses_lm_ieks = np.zeros((num_mc_samples, num_iter))
    neeses_gn_ipls = np.zeros((num_mc_samples, num_iter))
    neeses_lm_ipls = np.zeros((num_mc_samples, num_iter))

    for mc_iter in range(num_mc_samples):
        log.info(f"MC iter: {mc_iter+1}/{num_mc_samples}")
        states, measurements = simulate_data(motion_model, meas_model, prior_mean[:-1], time_steps=500)
        measurements = measurements[:, :2]
        cost_fn_eks = partial(
            analytical_smoothing_cost,
            measurements=measurements,
            m_1_0=prior_mean,
            P_1_0=prior_cov,
            motion_model=motion_model,
            meas_model=meas_model,
        )

        sigma_point_method = SphericalCubature()
        cost_fn_ipls = partial(
            slr_smoothing_cost_pre_comp,
            measurements=measurements,
            m_1_0=prior_mean,
            P_1_0=prior_cov,
        )

        ms_gn_ieks, Ps_gn_ieks, cost_gn_ieks, tmp_rmse, tmp_nees = run_smoothing(
            Ieks(motion_model, meas_model, num_iter), states, measurements, prior_mean, prior_cov, cost_fn_eks
        )
        rmses_gn_ieks[mc_iter, :] = tmp_rmse
        neeses_gn_ieks[mc_iter, :] = tmp_nees

        ms_lm_ieks, Ps_lm_ieks, cost_lm_ieks, tmp_rmse, tmp_nees = run_smoothing(
            LmIeks(motion_model, meas_model, num_iter, cost_improv_iter_lim=10, lambda_=1e-2, nu=10),
            states,
            measurements,
            prior_mean,
            prior_cov,
            cost_fn_eks,
        )
        rmses_lm_ieks[mc_iter, :] = tmp_rmse
        neeses_lm_ieks[mc_iter, :] = tmp_nees

        ms_gn_ipls, Ps_gn_ipls, cost_gn_ipls, tmp_rmse, tmp_nees = run_smoothing(
            SigmaPointIpls(motion_model, meas_model, sigma_point_method, num_iter),
            states,
            measurements,
            prior_mean,
            prior_cov,
            slr_noop_cost,
        )
        rmses_gn_ipls[mc_iter, :] = tmp_rmse
        neeses_gn_ipls[mc_iter, :] = tmp_nees

        ms_lm_ipls, Ps_lm_ipls, cost_lm_ipls, tmp_rmse, tmp_nees = run_smoothing(
            SigmaPointLmIpls(
                motion_model, meas_model, sigma_point_method, num_iter, cost_improv_iter_lim=10, lambda_=1e-4, nu=10
            ),
            states,
            measurements,
            prior_mean,
            prior_cov,
            cost_fn_ipls,
        )
        rmses_lm_ipls[mc_iter, :] = tmp_rmse
        neeses_lm_ipls[mc_iter, :] = tmp_nees

    label_gn_ieks, label_lm_ieks, label_gn_ipls, label_lm_ipls = "GN-IEKS", "LM-IEKS", "GN-IPLS", "LM-IPLS"
    rmse_stats = [
        (rmses_gn_ieks, label_gn_ieks),
        (rmses_lm_ieks, label_lm_ieks),
        (rmses_gn_ipls, label_gn_ipls),
        (rmses_lm_ipls, label_lm_ipls),
    ]

    nees_stats = [
        (neeses_gn_ieks, label_gn_ieks),
        (neeses_lm_ieks, label_lm_ieks),
        (neeses_gn_ipls, label_gn_ipls),
        (neeses_lm_ipls, label_lm_ipls),
    ]

    save_stats(Path.cwd() / "results", "RMSE", rmse_stats)
    save_stats(Path.cwd() / "results", "NEES", nees_stats)
    # tikz_stats(Path.cwd().parent / "paper/fig/ct_bearings_only_metrics/", "RMSE", rmse_stats)
    # tikz_stats(Path.cwd().parent / "paper/fig/ct_bearings_only_metrics/", "NEES", nees_stats)
    tikz_stats(Path.cwd() / "tmp_results", "RMSE", rmse_stats)
    tikz_stats(Path.cwd() / "tmp_results", "NEES", nees_stats)
    plot_stats(rmse_stats, "RMSE")
    plot_stats(nees_stats, "NEES")


def tikz_stats(dir_, name, stats):
    (dir_ / name.lower()).mkdir(parents=True, exist_ok=True)
    num_iter = stats[0][0].shape[1]
    iter_range = np.arange(1, num_iter + 1)
    stats = [(mc_stats(stat_), label) for stat_, label in stats]
    for (mean, err), label in stats:
        write_to_tikz_file(tikz_err_bar_tab_format(iter_range, mean, err), dir_ / name.lower(), f"{label.lower()}.data")


def save_stats(res_dir: Path, name: str, stats):
    (res_dir / name.lower()).mkdir(parents=True, exist_ok=True)
    for stat, label in stats:
        np.savetxt(res_dir / name.lower() / f"{label.lower()}.csv", stat)


def plot_stats(stats, title):
    num_iter = stats[0][0].shape[1]
    stats = [(mc_stats(stat_), label) for stat_, label in stats]
    fig, ax = plt.subplots()
    for (mean, err), label in stats:
        ax.errorbar(x=np.arange(1, num_iter + 1), y=mean, yerr=err, label=label)
    ax.set_title(title)
    ax.legend()
    plt.show()


def plot_metrics(costs, rmses, neeses, exp_name, tikz_dir):
    iter_ticks = np.arange(1, len(costs[0][1]) + 1)
    fig, (cost_ax, rmse_ax, nees_ax) = plt.subplots(3)
    for label, cost in costs:
        print(label, cost)
        cost_ax.plot(iter_ticks, cost, label=label)
    cost_ax.set_title("Cost")
    cost_ax.legend()
    for label, rmse_ in rmses:
        rmse_ax.plot(iter_ticks, rmse_, label=label)
    rmse_ax.set_title("RMSE")
    rmse_ax.legend()
    for label, nees_ in neeses:
        nees_ax.plot(iter_ticks, nees_, label=label)
    nees_ax.set_title("NEES")
    rmse_ax.legend()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="LM-IEKS paper experiment.")
    parser.add_argument("--meas_type", type=MeasType, required=True)
    parser.add_argument("--num_iter", type=int, default=10)
    parser.add_argument("--num_mc_samples", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    main()
