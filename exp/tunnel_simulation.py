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
from src.models.coord_turn import CoordTurn
from data.lm_ieks_paper.coord_turn_example import simulate_data
from src.smoother.slr.lm_ipls import SigmaPointLmIpls
from src.slr.sigma_points import SigmaPointSlr
from src.sigma_points import SphericalCubature
from src.cost import analytical_smoothing_cost, slr_smoothing_cost, slr_smoothing_cost_pre_comp
from exp.lm_ieks_paper import plot_results, plot_cost
from src.analytics import rmse, nees
from src.models.range_bearing import to_cartesian_coords
from data.tunnel_traj import get_states_and_meas
from exp.coord_turn_bearings_only import calc_iter_metrics


def main():
    log = logging.getLogger(__name__)
    experiment_name = "tunnel_simulation"
    setup_logger(f"logs/{experiment_name}.log", logging.DEBUG)
    log.info(f"Running experiment: {experiment_name}")

    np.random.seed(2)
    num_iter = 3

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
    noise_factor = 8
    sigma_r = 2 * noise_factor
    sigma_phi = noise_factor * 0.5 * np.pi / 180

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
    lm_reg = 1e-2

    results = []
    cost_fn_eks = partial(
        analytical_smoothing_cost,
        meas=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
        motion_model=motion_model,
        meas_model=meas_model,
    )

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

    cost_fn_ipls = partial(
        slr_smoothing_cost_pre_comp,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
    )
    # ms_gn_ieks, Ps_gn_ieks, cost_gn_ieks, tmp_rmse, tmp_nees = run_smoothing(
    #     Ieks(motion_model, meas_model, num_iter), states, measurements, prior_mean, prior_cov, cost_fn_eks
    # )
    # results.append((ms_gn_ieks, Ps_gn_ieks, cost_gn_ieks[1:], "GN-IEKS"))

    # ms_lm_ieks, Ps_lm_ieks, cost_lm_ieks, tmp_rmse, tmp_nees = run_smoothing(
    #     LmIeks(motion_model, meas_model, num_iter, cost_improv_iter_lim=10, lambda_=lm_reg, nu=10),
    #     states,
    #     measurements,
    #     prior_mean,
    #     prior_cov,
    #     cost_fn_eks,
    # )
    # results.append((ms_lm_ieks, Ps_lm_ieks, cost_lm_ieks[1:], "LM-IEKS"))

    # ms_gn_ipls, Ps_gn_ipls, cost_gn_ipls, rmses_gn_ipls, neeses_gn_ipls = run_smoothing(
    #     SigmaPointIpls(motion_model, meas_model, sigma_point_method, num_iter),
    #     states,
    #     measurements,
    #     prior_mean,
    #     prior_cov,
    #     cost_fn_ipls,
    # )
    # results.append((ms_gn_ipls, Ps_gn_ipls, cost_gn_ipls[1:], "GN-IPLS"))
    ms_lm_ipls, Ps_lm_ipls, cost_lm_ipls, rmses_lm_ipls, neeses_lm_ipls = run_smoothing(
        SigmaPointLmIpls(
            motion_model, meas_model, sigma_point_method, num_iter, cost_improv_iter_lim=10, lambda_=lm_reg, nu=10
        ),
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
        cartes_meas,
        skip_cov=10,
    )
    # plot_metrics(
    #     [
    #         (cost_gn_ieks[1:], "GN-IEKS"),
    #         (cost_lm_ieks[1:], "LM-IEKS"),
    #         (cost_gn_ipls[1:], "GN-IPLS"),
    #         (cost_lm_ipls[1:], "LM-IPLS"),
    #     ],
    #     [
    #         (rmses_gn_ieks, "GN-IEKS"),
    #         (rmses_lm_ieks, "LM-IEKS"),
    #         (rmses_gn_ipls, "LM-IPLS"),
    #         (rmses_lm_ipls, "LM-IPLS"),
    #     ],
    #     [
    #         (neeses_gn_ieks, "GN-IEKS"),
    #         (neeses_lm_ieks, "LM-IEKS"),
    #         (neeses_gn_ipls, "LM-IPLS"),
    #         (neeses_lm_ipls, "LM-IPLS"),
    #     ],
    # )


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


if __name__ == "__main__":
    main()
