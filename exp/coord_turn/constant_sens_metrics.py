"""Example:
Reproducing the experiment in the paper:
"Levenberg-Marquardt and line-search extended Kalman smoother".
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
from src.smoother.ext.ls_ieks import LsIeks
from src.smoother.slr.ipls import SigmaPointIpls
from src.smoother.slr.lm_ipls import SigmaPointLmIpls
from src.smoother.slr.ls_ipls import SigmaPointLsIpls
from src.line_search import GridSearch
from src.utils import setup_logger, tikz_err_bar_tab_format, tikz_stats, save_stats
from src.models.range_bearing import MultiSensorBearings, MultiSensorRange
from src.models.coord_turn import CoordTurn
from src.slr.sigma_points import SigmaPointSlr
from src.sigma_points import SphericalCubature
from src.cost_fn.ext import analytical_smoothing_cost
from src.cost_fn.slr import slr_smoothing_cost_pre_comp, slr_smoothing_cost_means
from src.analytics import rmse, nees
from src.visualization import to_tikz, write_to_tikz_file, plot_scalar_metric_err_bar
from data.lm_ieks_paper.coord_turn_example import simulate_data, save_states_and_meas
from exp.coord_turn.common import MeasType, run_smoothing, mc_stats, calc_iter_metrics, plot_results


def main():
    log = logging.getLogger(__name__)
    args = parse_args()
    experiment_name = "ct_constant_sens"
    setup_logger(f"logs/{experiment_name}.log", logging.INFO)
    log.info(f"Running experiment: {experiment_name}")
    if not args.random:
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

    lambda_ = 1e-2
    nu = 10

    num_iter = args.num_iter
    if args.meas_type == MeasType.Range:
        meas_model = MultiSensorRange(sensors, R)
        log.info("Using measurement model: MultiSensorRange")
    elif args.meas_type == MeasType.Bearings:
        meas_model = MultiSensorBearings(sensors, R)
        log.info("Using measurement model: MultiSensorBearings")

    states, measurements = simulate_data(motion_model, meas_model, prior_mean[:-1], time_steps=500)

    num_mc_samples = args.num_mc_samples

    rmses_ieks = np.zeros((num_mc_samples, num_iter))
    rmses_lm_ieks = np.zeros((num_mc_samples, num_iter))
    rmses_ls_ieks = np.zeros((num_mc_samples, num_iter))
    rmses_ipls = np.zeros((num_mc_samples, num_iter))
    rmses_lm_ipls = np.zeros((num_mc_samples, num_iter))
    rmses_ls_ipls = np.zeros((num_mc_samples, num_iter))

    neeses_ieks = np.zeros((num_mc_samples, num_iter))
    neeses_lm_ieks = np.zeros((num_mc_samples, num_iter))
    neeses_ls_ieks = np.zeros((num_mc_samples, num_iter))
    neeses_ipls = np.zeros((num_mc_samples, num_iter))
    neeses_lm_ipls = np.zeros((num_mc_samples, num_iter))
    neeses_ls_ipls = np.zeros((num_mc_samples, num_iter))

    grid_search_points = 10

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

        ms_ieks, Ps_ieks, cost_ieks, tmp_rmse, tmp_nees = run_smoothing(
            Ieks(motion_model, meas_model, num_iter), states, measurements, prior_mean, prior_cov, cost_fn_eks
        )
        rmses_ieks[mc_iter, :] = tmp_rmse
        neeses_ieks[mc_iter, :] = tmp_nees

        ms_lm_ieks, Ps_lm_ieks, cost_lm_ieks, tmp_rmse, tmp_nees = run_smoothing(
            LmIeks(motion_model, meas_model, num_iter, cost_improv_iter_lim=10, lambda_=lambda_, nu=nu),
            states,
            measurements,
            prior_mean,
            prior_cov,
            cost_fn_eks,
        )
        rmses_lm_ieks[mc_iter, :] = tmp_rmse
        neeses_lm_ieks[mc_iter, :] = tmp_nees

        ms_ls_ieks, Ps_ls_ieks, cost_ls_ieks, tmp_rmse, tmp_nees = run_smoothing(
            LsIeks(motion_model, meas_model, num_iter, GridSearch(cost_fn_eks, grid_search_points)),
            states,
            measurements,
            prior_mean,
            prior_cov,
            cost_fn_eks,
        )
        rmses_ls_ieks[mc_iter, :] = tmp_rmse
        neeses_ls_ieks[mc_iter, :] = tmp_nees

        sigma_point_method = SphericalCubature()
        cost_fn_ipls = partial(
            slr_smoothing_cost_pre_comp, measurements=measurements, m_1_0=prior_mean, P_1_0_inv=np.linalg.inv(prior_cov)
        )

        ms_ipls, Ps_ipls, cost_ipls, tmp_rmse, tmp_nees = run_smoothing(
            SigmaPointIpls(motion_model, meas_model, sigma_point_method, num_iter),
            states,
            measurements,
            prior_mean,
            prior_cov,
            None,
        )
        rmses_ipls[mc_iter, :] = tmp_rmse
        neeses_ipls[mc_iter, :] = tmp_nees

        ms_lm_ipls, Ps_lm_ipls, cost_lm_ipls, tmp_rmse, tmp_nees = run_smoothing(
            SigmaPointLmIpls(
                motion_model, meas_model, sigma_point_method, num_iter, cost_improv_iter_lim=10, lambda_=lambda_, nu=nu
            ),
            states,
            measurements,
            prior_mean,
            prior_cov,
            cost_fn_ipls,
        )
        rmses_lm_ipls[mc_iter, :] = tmp_rmse
        neeses_lm_ipls[mc_iter, :] = tmp_nees

        ls_cost_fn = partial(
            slr_smoothing_cost_means,
            measurements=measurements,
            m_1_0=prior_mean,
            P_1_0_inv=np.linalg.inv(prior_cov),
            motion_fn=motion_model.map_set,
            meas_fn=meas_model.map_set,
            slr_method=SigmaPointSlr(sigma_point_method),
        )
        ms_ls_ipls, Ps_ls_ipls, cost_ls_ipls, tmp_rmse, tmp_nees = run_smoothing(
            SigmaPointLsIpls(motion_model, meas_model, sigma_point_method, num_iter, GridSearch, grid_search_points),
            states,
            measurements,
            prior_mean,
            prior_cov,
            ls_cost_fn,
        )
        rmses_ls_ipls[mc_iter, :] = tmp_rmse
        neeses_ls_ipls[mc_iter, :] = tmp_nees

    label_ieks, label_lm_ieks, label_ls_ieks, label_ipls, label_lm_ipls, label_ls_ipls = (
        "IEKS",
        "LM-IEKS",
        "LS-IEKS",
        "IPLS",
        "LM-IPLS",
        "LS-IPLS",
    )
    rmse_stats = [
        (rmses_ieks, label_ieks),
        (rmses_lm_ieks, label_lm_ieks),
        (rmses_ls_ieks, label_ls_ieks),
        (rmses_ipls, label_ipls),
        (rmses_lm_ipls, label_lm_ipls),
        (rmses_ls_ipls, label_ls_ipls),
    ]

    nees_stats = [
        (neeses_ieks, label_ieks),
        (neeses_lm_ieks, label_lm_ieks),
        (neeses_ls_ieks, label_ls_ieks),
        (neeses_ipls, label_ipls),
        (neeses_lm_ipls, label_lm_ipls),
        (neeses_ls_ipls, label_ls_ipls),
    ]

    save_stats(Path.cwd() / "results" / experiment_name, "RMSE", rmse_stats)
    save_stats(Path.cwd() / "results" / experiment_name, "NEES", nees_stats)
    # tikz_stats(Path.cwd().parent / "paper/fig/ct_bearings_only_metrics/", "RMSE", rmse_stats)
    # tikz_stats(Path.cwd().parent / "paper/fig/ct_bearings_only_metrics/", "NEES", nees_stats)
    tikz_stats(Path.cwd() / "tmp_results" / experiment_name, "RMSE", rmse_stats)
    tikz_stats(Path.cwd() / "tmp_results" / experiment_name, "NEES", nees_stats)
    plot_scalar_metric_err_bar(rmse_stats, "RMSE")
    plot_scalar_metric_err_bar(nees_stats, "NEES")


def parse_args():
    parser = argparse.ArgumentParser(description="LM-IEKS paper experiment.")
    parser.add_argument("--meas_type", type=MeasType, required=True)
    parser.add_argument("--num_iter", type=int, default=10)
    parser.add_argument("--num_mc_samples", type=int, default=100)
    parser.add_argument("--random", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main()
