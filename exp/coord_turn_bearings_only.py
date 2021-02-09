"""Example:
Reproducing the experiment in the paper:
"Levenberg-Marquardt and line-search extended Kalman smoother".

The exact realisation of the experiment in the paper is found in
`exp/lm_ieks_paper.py`
"""

import logging
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from src import visualization as vis
from src.filter.ekf import Ekf
from src.smoother.ext.eks import Eks
from src.smoother.ext.ieks import Ieks
from src.smoother.ext.lm_ieks import LmIeks
from src.smoother.slr.ipls import SigmaPointIpls
from src.utils import setup_logger
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import LmCoordTurn
from data.lm_ieks_paper.coord_turn_example import simulate_data
from src.smoother.slr.reg_ipls import SigmaPointRegIpls
from src.slr.sigma_points import SigmaPointSlr
from src.sigma_points import SphericalCubature
from src.cost import analytical_smoothing_cost, slr_smoothing_cost
from exp.lm_ieks_paper import plot_results, plot_cost
from src.analytics import rmse, nees


def main():
    log = logging.getLogger(__name__)
    experiment_name = "coord_turn_bearings_only"
    setup_logger(f"logs/{experiment_name}.log", logging.INFO)
    log.info(f"Running experiment: {experiment_name}")
    # seed = 0
    # np.random.seed(seed)
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
    motion_model = LmCoordTurn(dt, Q)

    sens_pos_1 = np.array([-1.5, 0.5])
    sens_pos_2 = np.array([1, 1])
    sensors = np.row_stack((sens_pos_1, sens_pos_2))
    std = 0.5
    R = std ** 2 * np.eye(2)
    meas_model = MultiSensorRange(sensors, R)

    prior_mean = np.array([0, 0, 1, 0, 0])
    prior_cov = np.diag([0.1, 0.1, 1, 1, 1])

    states, measurements = simulate_data(sens_pos_1, sens_pos_2, std, dt, prior_mean[:-1], time_steps=500)

    num_iter = 3

    results = []
    cost_fn_eks = partial(
        analytical_smoothing_cost,
        meas=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
        motion_model=motion_model,
        meas_model=meas_model,
    )
    ms_gn_ieks, Ps_gn_ieks, cost_gn_ieks, rmses_gn_ieks, neeses_gn_ieks = gn_ieks(
        motion_model, meas_model, num_iter, states, measurements, prior_mean, prior_cov, cost_fn_eks
    )
    results.append((ms_gn_ieks, Ps_gn_ieks, cost_gn_ieks[1:], "GN-IEKS"))
    ms_lm_ieks, Ps_lm_ieks, cost_lm_ieks, rmses_lm_ieks, neeses_lm_ieks = lm_ieks(
        motion_model, meas_model, num_iter, states, measurements, prior_mean, prior_cov, cost_fn_eks
    )
    results.append((ms_lm_ieks, Ps_lm_ieks, cost_lm_ieks[1:], "LM-IEKS"))

    sigma_point_method = SphericalCubature()
    cost_fn_ipls = partial(
        slr_smoothing_cost,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
        motion_model=motion_model,
        meas_model=meas_model,
        slr=SigmaPointSlr(sigma_point_method),
    )
    ms_gn_ipls, Ps_gn_ipls, cost_gn_ipls, rmses_gn_ipls, neeses_gn_ipls = gn_ipls(
        motion_model,
        meas_model,
        sigma_point_method,
        num_iter,
        states,
        measurements,
        prior_mean,
        prior_cov,
        cost_fn_ipls,
    )
    results.append((ms_gn_ipls, Ps_gn_ipls, cost_gn_ipls[1:], "GN-IPLS"))
    ms_lm_ipls, Ps_lm_ipls, cost_lm_ipls, rmses_lm_ipls, neeses_lm_ipls = lm_ipls(
        motion_model,
        meas_model,
        sigma_point_method,
        num_iter,
        states,
        measurements,
        prior_mean,
        prior_cov,
        cost_fn_ipls,
    )
    results.append((ms_lm_ipls, Ps_lm_ipls, cost_lm_ipls[1:], "LM-IPLS"))
    plot_results(
        states,
        results,
    )
    plot_metrics(
        [
            (cost_gn_ieks[1:], "GN-IEKS"),
            (cost_lm_ieks[1:], "LM-IEKS"),
            (cost_gn_ipls[1:], "GN-IPLS"),
            (cost_lm_ipls[1:], "LM-IPLS"),
        ],
        [
            (rmses_gn_ieks, "GN-IEKS"),
            (rmses_lm_ieks, "LM-IEKS"),
            (rmses_gn_ipls, "LM-IPLS"),
            (rmses_lm_ipls, "LM-IPLS"),
        ],
        [
            (neeses_gn_ieks, "GN-IEKS"),
            (neeses_lm_ieks, "LM-IEKS"),
            (neeses_gn_ipls, "LM-IPLS"),
            (neeses_lm_ipls, "LM-IPLS"),
        ],
    )


def plot_metrics(costs, rmses, neeses):
    iter_ticks = np.arange(1, len(costs[0][0]) + 1)
    fig, (cost_ax, rmse_ax, nees_ax) = plt.subplots(3)
    for cost, label in costs:
        cost_ax.plot(iter_ticks, cost, label=label)
    cost_ax.set_title("Cost")
    cost_ax.legend()
    for rmse_, label in rmses:
        rmse_ax.plot(iter_ticks, rmse_, label=label)
    rmse_ax.set_title("RMSE")
    rmse_ax.legend()
    for nees_, label in neeses:
        nees_ax.plot(iter_ticks, nees_, label=label)
    rmse_ax.set_title("NEES")
    rmse_ax.legend()
    plt.show()


def calc_iter_metrics(metric_fn, estimates, states):
    return np.array([metric_fn(means, covs, states) for means, covs in estimates])


def gn_ieks(motion_model, meas_model, num_iter, states, measurements, prior_mean, prior_cov, cost_fn):
    smoother = Ieks(motion_model, meas_model, num_iter)
    _, _, ms, Ps, iter_cost = smoother.filter_and_smooth(measurements, prior_mean, prior_cov, cost_fn)
    rmses = calc_iter_metrics(
        lambda means, covs, states: rmse(means[:, :-1], states), smoother.stored_estimates(), states
    )
    neeses = calc_iter_metrics(
        lambda means, covs, states: np.mean(nees(means[:, :-1], states, covs[:, :-1, :-1])),
        smoother.stored_estimates(),
        states,
    )
    return ms, Ps, iter_cost, rmses, neeses


def lm_ieks(motion_model, meas_model, num_iter, states, measurements, prior_mean, prior_cov, cost_fn):
    cost_improv_iter_lim = 10
    lambda_ = 1e-2
    nu = 10
    smoother = LmIeks(motion_model, meas_model, num_iter, cost_improv_iter_lim, lambda_, nu)
    _, _, ms, Ps, iter_cost = smoother.filter_and_smooth(measurements, prior_mean, prior_cov, cost_fn)
    rmses = calc_iter_metrics(
        lambda means, covs, states: rmse(means[:, :-1], states), smoother.stored_estimates(), states
    )
    neeses = calc_iter_metrics(
        lambda means, covs, states: np.mean(nees(means[:, :-1], states, covs[:, :-1, :-1])),
        smoother.stored_estimates(),
        states,
    )
    return ms, Ps, iter_cost, rmses, neeses


def gn_ipls(
    motion_model, meas_model, sigma_point_method, num_iter, states, measurements, prior_mean, prior_cov, cost_fn
):
    smoother = SigmaPointIpls(motion_model, meas_model, sigma_point_method, num_iter)
    _, _, ms, Ps, iter_cost = smoother.filter_and_smooth(measurements, prior_mean, prior_cov, cost_fn)
    rmses = calc_iter_metrics(
        lambda means, covs, states: rmse(means[:, :-1], states), smoother.stored_estimates(), states
    )
    neeses = calc_iter_metrics(
        lambda means, covs, states: np.mean(nees(means[:, :-1], states, covs[:, :-1, :-1])),
        smoother.stored_estimates(),
        states,
    )
    return ms, Ps, iter_cost, rmses, neeses


def lm_ipls(
    motion_model, meas_model, sigma_point_method, num_iter, states, measurements, prior_mean, prior_cov, cost_fn
):
    lambda_ = 1e-2
    nu = 10
    cost_improv_iter_lim = 10
    smoother = SigmaPointRegIpls(
        motion_model, meas_model, sigma_point_method, num_iter, cost_improv_iter_lim, lambda_, nu
    )
    _, _, ms, Ps, iter_cost = smoother.filter_and_smooth(measurements, prior_mean, prior_cov, cost_fn)
    rmses = calc_iter_metrics(
        lambda means, covs, states: rmse(means[:, :-1], states), smoother.stored_estimates(), states
    )
    neeses = calc_iter_metrics(
        lambda means, covs, states: np.mean(nees(means[:, :-1], states, covs[:, :-1, :-1])),
        smoother.stored_estimates(),
        states,
    )
    return ms, Ps, iter_cost, rmses, neeses


if __name__ == "__main__":
    main()
