"""Example: Compute metrics for CT model with varying sensors"""

import logging
import argparse
from pathlib import Path
from functools import partial
import numpy as np
from src.smoother.ext.ieks import Ieks
from src.smoother.ext.lm_ieks import LmIeks
from src.smoother.ext.ls_ieks import LsIeks
from src.smoother.slr.ipls import SigmaPointIpls
from src.smoother.slr.lm_ipls import SigmaPointLmIpls
from src.smoother.slr.ls_ipls import SigmaPointLsIpls
from src.slr.sigma_points import SigmaPointSlr
from src.sigma_points import SphericalCubature
from src.line_search import ArmijoWolfeLineSearch
from src.cost_fn.ext import analytical_smoothing_cost_time_dep, dir_der_analytical_smoothing_cost
from src.cost_fn.slr import slr_smoothing_cost_pre_comp, slr_smoothing_cost_means
from src.utils import setup_logger, save_stats
from src.visualization import plot_scalar_metric_err_bar
from src.models.range_bearing import BearingsVaryingSensors, MultiSensorBearings
from src.models.coord_turn import CoordTurn
from data.lm_ieks_paper.coord_turn_example import Type, get_specific_states_from_file, simulate_data
from exp.coord_turn.common import run_smoothing, modify_meas


def main():
    args = parse_args()
    log = logging.getLogger(__name__)
    experiment_name = "ct_varying_sens_metrics.py"
    setup_logger(f"logs/{experiment_name}.log", logging.INFO)
    log.info(f"Running experiment: {experiment_name}")
    if not args.random:
        np.random.seed(0)

    dt = 0.01
    qc = 0.01
    qw = 10
    prior_mean = np.array([0, 0, 1, 0, 0])
    prior_cov = np.diag([0.1, 0.1, 1, 1, 1])
    D_x = prior_mean.shape[0]
    K = 500
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
    std = 0.5
    R_uncertain = std ** 2 * np.eye(2)

    double_sensors = np.row_stack((sens_pos_1, sens_pos_2))
    double_meas_model = MultiSensorBearings(double_sensors, R_uncertain)

    single_sensor = sens_pos_2.reshape((1, 2))
    std = 0.001
    R_certain = std ** 2 * np.eye(1)
    single_meas_model = MultiSensorBearings(single_sensor, R_certain)
    # Create a set of time steps where the original two sensor measurements are replaced with single ones.
    # single_meas_time_steps = set(list(range(0, 100, 5))[1:])
    single_meas_time_steps = set(list(range(0, K, 50))[1:])
    meas_model = BearingsVaryingSensors(double_meas_model, single_meas_model, single_meas_time_steps)

    num_iter = args.num_iter
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

    # LM parameters
    lambda_ = 1e-2
    nu = 10
    # Armijo-Wolfe parameters
    c_1, c_2 = 0.1, 0.9

    D_x = prior_mean.shape[0]
    K = 500
    init_traj = (np.zeros((K, D_x)), np.array(K * [prior_cov]))
    for mc_iter in range(num_mc_samples):
        log.info(f"MC iter: {mc_iter+1}/{num_mc_samples}")
        if args.random:
            states, measurements = simulate_data(motion_model, double_meas_model, prior_mean[:-1], time_steps=K)
        else:
            states, all_meas, _, xs_ss = get_specific_states_from_file(
                Path.cwd() / "data/lm_ieks_paper", Type.LM, num_iter
            )
            measurements = all_meas[:, 2:]

        # Change measurments so that some come from the alternative model.
        measurements = modify_meas(measurements, states, meas_model, True)

        cost_fn_eks = partial(
            analytical_smoothing_cost_time_dep,
            measurements=measurements,
            m_1_0=prior_mean,
            P_1_0=prior_cov,
            motion_model=motion_model,
            meas_model=meas_model,
        )

        ms_ieks, Ps_ieks, cost_ieks, tmp_rmse, tmp_nees = run_smoothing(
            Ieks(motion_model, meas_model, num_iter),
            states,
            measurements,
            prior_mean,
            prior_cov,
            cost_fn_eks,
            init_traj,
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
            init_traj,
        )
        rmses_lm_ieks[mc_iter, :] = tmp_rmse
        neeses_lm_ieks[mc_iter, :] = tmp_nees

        dir_der_eks = partial(
            dir_der_analytical_smoothing_cost,
            measurements=measurements,
            m_1_0=prior_mean,
            P_1_0=prior_cov,
            motion_model=motion_model,
            meas_model=meas_model,
        )
        ms_ls_ieks, Ps_ls_ieks, cost_ls_ieks, tmp_rmse, tmp_nees = run_smoothing(
            LsIeks(
                motion_model, meas_model, num_iter, ArmijoWolfeLineSearch(cost_fn_eks, dir_der_eks, c_1=c_1, c_2=c_1)
            ),
            states,
            measurements,
            prior_mean,
            prior_cov,
            cost_fn_eks,
            init_traj,
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
            init_traj,
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
            init_traj,
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
            SigmaPointLsIpls(
                motion_model, meas_model, sigma_point_method, num_iter, partial(ArmijoWolfeLineSearch, c_1=c_1, c_2=c_2)
            ),
            states,
            measurements,
            prior_mean,
            prior_cov,
            ls_cost_fn,
            init_traj,
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
    plot_scalar_metric_err_bar(rmse_stats, "RMSE")
    plot_scalar_metric_err_bar(nees_stats, "NEES")


def parse_args():
    parser = argparse.ArgumentParser(description="CT experiment with varying meas. model.")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--num_iter", type=int, default=10)
    parser.add_argument("--num_mc_samples", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    main()
