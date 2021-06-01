"""Example: Levenberg-Marquardt regularised IEKS smoothing

Reproducing the experiment (with two different meas. models) in the paper:

"Levenberg-Marquardt and line-search extended Kalman smoother".

The problem is evaluated for
(Original experiment)
- IEKS
- LM-IEKS
(Additional models)
- IPLS
- LM-IPLS
"""

from enum import Enum
import logging
import argparse
from pathlib import Path
from functools import partial
import numpy as np
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
from src.slr.sigma_points import SigmaPointSlr
from src.sigma_points import SphericalCubature
from src.cost import analytical_smoothing_cost_time_dep, slr_smoothing_cost_means, slr_smoothing_cost_pre_comp
from src.utils import setup_logger, tikz_err_bar_tab_format, tikz_stats, save_stats
from src.analytics import rmse, mc_stats
from src.visualization import to_tikz, write_to_tikz_file, plot_scalar_metric_err_bar
from src.models.range_bearing import BearingsVaryingSensors, MultiSensorBearings
from src.models.coord_turn import CoordTurn
from data.lm_ieks_paper.coord_turn_example import Type, get_specific_states_from_file, simulate_data
from exp.coord_turn.common import run_smoothing, calc_iter_metrics, modify_meas


def main():
    args = parse_args()
    log = logging.getLogger(__name__)
    experiment_name = "tmp_ct_varying_sens"
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

    lambda_ = 1e-2
    nu = 10

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

        sigma_point_method = SphericalCubature()
        cost_fn_ipls = partial(
            slr_smoothing_cost_pre_comp, measurements=measurements, m_1_0=prior_mean, P_1_0_inv=np.linalg.inv(prior_cov)
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

        ms_ls_ieks, Ps_ls_ieks, cost_ls_ieks, tmp_rmse, tmp_nees = run_smoothing(
            LsIeks(motion_model, meas_model, num_iter, GridSearch(cost_fn_eks, 20)),
            states,
            measurements,
            prior_mean,
            prior_cov,
            cost_fn_eks,
            init_traj,
        )
        rmses_ls_ieks[mc_iter, :] = tmp_rmse
        neeses_ls_ieks[mc_iter, :] = tmp_nees

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
            SigmaPointLsIpls(motion_model, meas_model, sigma_point_method, num_iter, GridSearch, 10),
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
    tikz_stats(Path.cwd() / "tmp_results", "RMSE", rmse_stats)
    tikz_stats(Path.cwd() / "tmp_results", "NEES", nees_stats)
    plot_scalar_metric_err_bar(rmse_stats, "RMSE")
    plot_scalar_metric_err_bar(nees_stats, "NEES")


# TODO: Sep module?
def plot_results(states, trajs_and_costs, meas, skip_cov=50):
    means_and_covs = [(ms, Ps, f"{label}-{len(cost)}") for (ms, Ps, cost, label) in trajs_and_costs]
    # costs = [
    #     (cost, f"{label}-{len(cost)}") for (_, _, cost, label) in trajs_and_costs
    # ]
    # _, (ax_1, ax_2) = plt.subplots(1, 2)
    fig, ax_1 = plt.subplots()
    vis.plot_2d_est(
        states,
        meas=meas,
        means_and_covs=means_and_covs,
        sigma_level=2,
        skip_cov=skip_cov,
        ax=ax_1,
    )
    write_to_tikz_file(to_tikz(fig), Path.cwd(), "aba.tikz")
    plt.show()


def plot_cost(ax, costs):
    for (iter_cost, label) in costs:
        ax.semilogy(iter_cost, label=label)
    ax.set_xlabel("Iteration number i")
    ax.set_ylabel("$L_{LM}$")
    ax.set_title("Cost function")


class MeasType(Enum):
    Range = "range"
    Bearings = "bearings"


def parse_args():
    parser = argparse.ArgumentParser(description="LM-IEKS paper experiment.")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--num_iter", type=int, default=10)
    parser.add_argument("--num_mc_samples", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    main()
