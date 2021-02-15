"""Experiment: Tunnel simulation

Testing robustness of smoothers in a simulated scenario
where a car passes through a tunnel, thereby going through the stages of

- Starting with low noise measurements (before the tunnel)
- Increased uncertainty while in the tunnel
(- Ending past the tunnel, again with certain measurements)
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
from src.models.range_bearing import RangeBearing
from src.models.coord_turn import LmCoordTurn
from data.lm_ieks_paper.coord_turn_example import simulate_data
from src.smoother.slr.reg_ipls import SigmaPointRegIpls
from src.slr.sigma_points import SigmaPointSlr
from src.sigma_points import SphericalCubature
from src.cost import analytical_smoothing_cost, slr_smoothing_cost
from exp.lm_ieks_paper import plot_results, plot_cost
from src.analytics import rmse, nees
from src.models.range_bearing import to_cartesian_coords
from src.cost import slr_smoothing_cost
from data.tunnel_traj import get_states_and_meas


def main():
    log = logging.getLogger(__name__)
    experiment_name = "tunnel_simulation"
    setup_logger(f"logs/{experiment_name}.log", logging.DEBUG)
    log.info(f"Running experiment: {experiment_name}")

    np.random.seed(2)
    num_iter = 3

    # Motion model
    sampling_period = 0.1
    v_scale = 2
    omega_scale = 2
    sigma_v = v_scale * 1
    sigma_omega = omega_scale * np.pi / 180
    Q = np.diag([0, 0, sampling_period * sigma_v ** 2, 0, sampling_period * sigma_omega ** 2])
    motion_model = LmCoordTurn(sampling_period, Q)

    # Meas model
    pos = np.array([100, -100])
    # sigma_r = 2
    # sigma_phi = 0.5 * np.pi / 180
    sigma_r = 4
    sigma_phi = 1 * np.pi / 180

    R = np.diag([sigma_r ** 2, sigma_phi ** 2])
    meas_model = RangeBearing(pos, R)

    # Generate data
    range_ = (0, None)
    tunnel_segment = [145, 165]
    # tunnel_segment = [None, None]
    states, measurements = get_states_and_meas(meas_model, R, range_, tunnel_segment)
    cartes_meas = np.apply_along_axis(partial(to_cartesian_coords, pos=pos), 1, measurements)

    prior_mean = np.array([0, 0, 1, 0, 0])
    prior_cov = np.diag([0.1, 0.1, 1, 1, 1])

    results = []
    cost_fn_eks = partial(
        analytical_smoothing_cost,
        meas=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
        motion_model=motion_model,
        meas_model=meas_model,
    )
    # ms_gn_ieks, Ps_gn_ieks, cost_gn_ieks, rmses_gn_ieks, neeses_gn_ieks = gn_ieks(
    #     motion_model, meas_model, num_iter, states, measurements, prior_mean, prior_cov, cost_fn_eks
    # )
    # results.append((ms_gn_ieks, Ps_gn_ieks, cost_gn_ieks[1:], "GN-IEKS"))
    # ms_lm_ieks, Ps_lm_ieks, cost_lm_ieks, rmses_lm_ieks, neeses_lm_ieks = lm_ieks(
    #     motion_model, meas_model, num_iter, states, measurements, prior_mean, prior_cov, cost_fn_eks
    # )
    # results.append((ms_lm_ieks, Ps_lm_ieks, cost_lm_ieks[1:], "LM-IEKS"))

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
        1,
        states,
        measurements,
        prior_mean,
        prior_cov,
        cost_fn_ipls,
    )
    results.append((ms_gn_ipls, Ps_gn_ipls, cost_gn_ipls[1:], "GN-IPLS"))
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
    # ms_lm_ipls, Ps_lm_ipls, cost_lm_ipls, rmses_lm_ipls, neeses_lm_ipls = lm_ipls(
    #     motion_model,
    #     meas_model,
    #     sigma_point_method,
    #     num_iter,
    #     states,
    #     measurements,
    #     prior_mean,
    #     prior_cov,
    #     cost_fn_ipls,
    # )
    # results.append((ms_lm_ipls, Ps_lm_ipls, cost_lm_ipls[1:], "LM-IPLS"))
    plot_results(
        states,
        results,
        cartes_meas,
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
        lambda means, covs, states: rmse(means[:, :2], states), smoother.stored_estimates(), states
    )
    neeses = calc_iter_metrics(
        lambda means, covs, states: np.mean(nees(means[:, :2], states, covs[:, :2, :2])),
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
        lambda means, covs, states: rmse(means[:, :2], states), smoother.stored_estimates(), states
    )
    neeses = calc_iter_metrics(
        lambda means, covs, states: np.mean(nees(means[:, :2], states, covs[:, :2, :2])),
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
        lambda means, covs, states: rmse(means[:, :2], states), smoother.stored_estimates(), states
    )
    neeses = calc_iter_metrics(
        lambda means, covs, states: np.mean(nees(means[:, :2], states, covs[:, :2, :2])),
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
        lambda means, covs, states: rmse(means[:, :2], states), smoother.stored_estimates(), states
    )
    neeses = calc_iter_metrics(
        lambda means, covs, states: np.mean(nees(means[:, :2], states, covs[:, :2, :2])),
        smoother.stored_estimates(),
        states,
    )
    return ms, Ps, iter_cost, rmses, neeses


if __name__ == "__main__":
    main()
