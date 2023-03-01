"""Experiment: Tunnel simulation

Testing robustness of smoothers in a simulated scenario
where a car passes through a tunnel, thereby going through the stages of

- Starting with low noise measurements (before the tunnel)
- Increased uncertainty while in the tunnel
(- Ending past the tunnel, again with certain measurements)
"""
import argparse
import logging
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from src.smoother.ext.ieks import Ieks
from src.smoother.ext.lm_ieks import LmIeks
from src.smoother.ext.ls_ieks import LsIeks
from src.smoother.slr.ipls import SigmaPointIpls
from src.smoother.slr.lm_ipls import SigmaPointLmIpls
from src.smoother.slr.ls_ipls import SigmaPointLsIpls
from src.line_search import ArmijoWolfeLineSearch
from src.cost_fn.ext import analytical_smoothing_cost, dir_der_analytical_smoothing_cost
from src.utils import setup_logger, save_stats
from src.models.range_bearing import RangeBearing
from src.models.coord_turn import CoordTurn
from src.slr.sigma_points import SigmaPointSlr
from src.sigma_points import SphericalCubature
from src.cost_fn.ext import analytical_smoothing_cost
from src.cost_fn.slr import slr_smoothing_cost_pre_comp, slr_smoothing_cost_means
from exp.coord_turn.common import plot_results, calc_iter_metrics
from src.analytics import rmse, nees
from data.tunnel_traj import get_states_and_meas
from src.visualization import plot_scalar_metric_err_bar
from pathlib import Path


def main():
    log = logging.getLogger(__name__)
    args = parse_args()
    experiment_name = "tunnel_simulation"
    setup_logger(f"logs/{experiment_name}.log", logging.INFO)
    log.info(f"Running experiment: {experiment_name}")

    np.random.seed(2)
    num_iter = args.num_iter

    # Motion model
    sampling_period = 0.1
    v_scale = 7
    omega_scale = 15
    sigma_v = v_scale * 1
    sigma_omega = omega_scale * np.pi / 180
    eps = 0.1
    Q = np.diag([eps, eps, sampling_period * sigma_v ** 2, eps, sampling_period * sigma_omega ** 2])
    motion_model = CoordTurn(sampling_period, Q)

    # Meas model
    pos = np.array([100, -100])
    # sigma_r = 2
    # sigma_phi = 0.5 * np.pi / 180
    noise_factor = 4
    sigma_r = 2 * noise_factor
    sigma_phi = noise_factor * 0.5 * np.pi / 180

    R = np.diag([sigma_r ** 2, sigma_phi ** 2])
    meas_model = RangeBearing(pos, R)

    # Generate data
    range_ = (0, None)
    tunnel_segment = [140, 175]
    # tunnel_segment = [None, None]
    prior_mean = np.array([0, 0, 1, 0, 0])
    prior_cov = np.diag([0.1, 0.1, 1, 1, 1])

    # LM parameters
    lambda_ = 1e-2
    nu = 10
    # Armijo-Wolfe parameters
    c_1, c_2 = 0.1, 0.9

    num_mc_samples = args.num_mc_samples
    rmses_ieks = np.zeros((num_mc_samples, num_iter))
    rmses_lm_ieks = np.zeros((num_mc_samples, num_iter))
    rmses_ls_ieks = np.zeros((num_mc_samples, num_iter))
    rmses_ipls = np.zeros((num_mc_samples, num_iter))
    rmses_lm_ipls = np.zeros((num_mc_samples, num_iter))
    rmses_ls_ipls = np.zeros((num_mc_samples, num_iter))

    neeses_gn_ieks = np.zeros((num_mc_samples, num_iter))
    neeses_lm_ieks = np.zeros((num_mc_samples, num_iter))
    neeses_ls_ieks = np.zeros((num_mc_samples, num_iter))
    neeses_gn_ipls = np.zeros((num_mc_samples, num_iter))
    neeses_lm_ipls = np.zeros((num_mc_samples, num_iter))
    neeses_ls_ipls = np.zeros((num_mc_samples, num_iter))
    for mc_iter in range(num_mc_samples):
        log.info(f"MC iter: {mc_iter+1}/{num_mc_samples}")
        states, measurements = get_states_and_meas(meas_model, R, range_, tunnel_segment)
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
            P_1_0_inv=np.linalg.inv(prior_cov),
        )
        ms_gn_ieks, Ps_gn_ieks, cost_gn_ieks, tmp_rmse, tmp_nees = run_smoothing(
            Ieks(motion_model, meas_model, num_iter), states, measurements, prior_mean, prior_cov, cost_fn_eks
        )
        rmses_ieks[mc_iter, :] = tmp_rmse
        neeses_gn_ieks[mc_iter, :] = tmp_nees

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
                motion_model,
                meas_model,
                num_iter,
                ArmijoWolfeLineSearch(cost_fn_eks, dir_der_eks, c_1=c_1, c_2=c_2),
            ),
            states,
            measurements,
            prior_mean,
            prior_cov,
            cost_fn_eks,
        )
        rmses_ls_ieks[mc_iter, :] = tmp_rmse
        neeses_ls_ieks[mc_iter, :] = tmp_nees

        ms_gn_ipls, Ps_gn_ipls, cost_gn_ipls, tmp_rmse, tmp_nees = run_smoothing(
            SigmaPointIpls(motion_model, meas_model, sigma_point_method, num_iter),
            states,
            measurements,
            prior_mean,
            prior_cov,
            None,
        )
        rmses_ipls[mc_iter, :] = tmp_rmse
        neeses_gn_ipls[mc_iter, :] = tmp_nees

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
            SigmaPointLsIpls(
                motion_model, meas_model, sigma_point_method, num_iter, partial(ArmijoWolfeLineSearch, c_1=c_1, c_2=c_2)
            ),
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
        (neeses_gn_ieks, label_ieks),
        (neeses_lm_ieks, label_lm_ieks),
        (neeses_ls_ieks, label_ls_ieks),
        (neeses_gn_ipls, label_ipls),
        (neeses_lm_ipls, label_lm_ipls),
        (neeses_ls_ipls, label_ls_ipls),
    ]

    save_stats(Path.cwd() / "results" / experiment_name, "RMSE", rmse_stats)
    save_stats(Path.cwd() / "results" / experiment_name, "NEES", nees_stats)
    plot_scalar_metric_err_bar(rmse_stats, "RMSE")
    plot_scalar_metric_err_bar(nees_stats, "NEES")


def run_smoothing(smoother, states, measurements, prior_mean, prior_cov, cost_fn, init_traj=None):
    """Common function that runs a smoother and collects metrics

    Some iterative smoothers may return early if they exceed the limit on the number of loss-improving trials.
    In those cases, the metrics are extended with the last element to a list of length `smoother.num_iter`
    """
    if init_traj is not None:
        _, _, ms, Ps, iter_cost = smoother.filter_and_smooth_with_init_traj(
            measurements, prior_mean, prior_cov, init_traj, 1, cost_fn
        )
        stored_est = smoother.stored_estimates()
        next(stored_est)
        stored_est = list(stored_est)
    else:
        _, _, ms, Ps, iter_cost = smoother.filter_and_smooth(measurements, prior_mean, prior_cov, cost_fn)
        stored_est = list(smoother.stored_estimates())
    rmses = calc_iter_metrics(
        lambda means, covs, states: rmse(means[:, :2], states), stored_est, states, smoother.num_iter
    )
    # assert np.allclose(ms_st, ms)
    neeses = calc_iter_metrics(
        lambda means, covs, states: np.mean(nees(states, means[:, :2], covs[:, :2, :2])),
        stored_est,
        states,
        smoother.num_iter,
    )
    return ms, Ps, iter_cost, rmses, neeses


def plot_metrics(costs, rmses, neeses):
    iter_ticks = np.arange(1, len(rmses[0][0]) + 1)
    fig, (cost_ax, rmse_ax, nees_ax) = plt.subplots(3)
    # for cost, label in costs:
    #     cost_ax.plot(iter_ticks, cost, label=label)
    # cost_ax.set_title("Cost")
    # cost_ax.legend()
    for rmse_, label in rmses:
        rmse_ax.plot(iter_ticks, rmse_, label=label)
    rmse_ax.set_title("RMSE")
    rmse_ax.legend()
    for nees_, label in neeses:
        nees_ax.plot(iter_ticks, nees_, label=label)
    rmse_ax.set_title("NEES")
    rmse_ax.legend()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="LM-IEKS paper experiment.")
    parser.add_argument("--num_iter", type=int, default=10)
    parser.add_argument("--num_mc_samples", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    main()
