"""Example: Levenberg-Marquardt regularised IPLS smoothing

Reproducing the experiment in the paper:

    "Levenberg-Marquardt and line-search extended Kalman smoother"

but with the Reg-IPLS rather than the LM-IEKS:
"""

import logging
from pathlib import Path
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from src import visualization as vis
from src.cost import slr_smoothing_cost, noop_cost
from src.smoother.slr.ipls import SigmaPointIpls
from src.smoother.slr.reg_ipls import SigmaPointRegIpls
from src.slr.sigma_points import SigmaPointSlr
from src.sigma_points import SphericalCubature
from src.utils import setup_logger
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import LmCoordTurn
from data.lm_ieks_paper.coord_turn_example import Type, get_specific_states_from_file


def main():
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    log = logging.getLogger(__name__)
    experiment_name = "reg_ipls"
    setup_logger(f"logs/{experiment_name}.log", logging.DEBUG)
    log.info(f"Running experiment: {experiment_name}")

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

    sigma_point_method = SphericalCubature()

    num_iter = 10
    states, measurements, _, _ = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.LM, num_iter)

    # This creates a prototype for the cost with arguments L(means, covs)
    # In the Smoother a proper cost function will be created as L(means) with fixed covs
    cost_fn = partial(
        slr_smoothing_cost,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
        motion_model=motion_model,
        meas_model=meas_model,
        slr=SigmaPointSlr(sigma_point_method),
    )
    # ms_gn, Ps_gn, cost_gn = gn_ipls(
    #     motion_model,
    #     meas_model,
    #     sigma_point_method,
    #     num_iter,
    #     measurements,
    #     prior_mean,
    #     prior_cov,
    #     cost_fn,
    # )
    # plot_results(states, [(ms_gn, Ps_gn, cost_gn[1:], "GN-IPLS")])
    ms_lm, Ps_lm, cost_lm = reg_ipls(
        motion_model, meas_model, sigma_point_method, num_iter, measurements, prior_mean, prior_cov, cost_fn
    )
    plot_results(
        states,
        [
            # (ms_gn, Ps_gn, cost_gn[1:], "GN-IPLS"),
            (ms_lm, Ps_lm, cost_lm[1:], "Reg-IPLS")
        ],
    )


def gn_ipls(motion_model, meas_model, sigma_point_method, num_iter, measurements, prior_mean, prior_cov, cost_fn):
    smoother = SigmaPointIpls(motion_model, meas_model, sigma_point_method, num_iter)
    # Note that the paper uses m_k = 0, k = 1, ..., K as the initial trajectory
    # This is the reason for not using the ordinary `filter_and_smooth` method.
    _, _, ms, Ps, iter_cost = smoother.filter_and_smooth(measurements, prior_mean, prior_cov, cost_fn)
    return ms, Ps, iter_cost


def reg_ipls(motion_model, meas_model, sigma_point_method, num_iter, measurements, prior_mean, prior_cov, cost_fn):
    lambda_ = 1e-2
    nu = 10
    cost_improv_iter_lim = 10
    smoother = SigmaPointRegIpls(
        motion_model, meas_model, sigma_point_method, num_iter, cost_improv_iter_lim, lambda_, nu
    )
    _, _, ms, Ps, iter_cost = smoother.filter_and_smooth(measurements, prior_mean, prior_cov, cost_fn)
    return ms, Ps, iter_cost


def plot_results(states, trajs_and_costs):
    means_and_covs = [(ms, Ps, f"{label}-{len(cost)}") for (ms, Ps, cost, label) in trajs_and_costs]
    costs = [(cost, f"{label}-{len(cost)}") for (_, _, cost, label) in trajs_and_costs]
    _, (ax_1, ax_2) = plt.subplots(1, 2)
    vis.plot_2d_est(
        states,
        meas=None,
        means_and_covs=means_and_covs,
        sigma_level=2,
        skip_cov=50,
        ax=ax_1,
    )
    plot_cost(ax_2, costs)
    plt.show()


def plot_cost(ax, costs):
    for (iter_cost, label) in costs:
        ax.semilogy(iter_cost, label=label)
    ax.set_xlabel("Iteration number i")
    ax.set_ylabel("$L_{LM}$")
    ax.set_title("Cost function")


if __name__ == "__main__":
    main()
