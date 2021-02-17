"""Example: Levenberg-Marquardt regularised IEKS smoothing

Reproducing the experiment in the paper:

"Levenberg-Marquardt and line-search extended Kalman smoother".

The problem is evaluated for
(Original experiment)
- (GN-)IEKS
- LM-IEKS
(Additional models)
- (GN-)IPLS
- Reg-IPLS (LM-IPLS)
"""

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
from src.smoother.slr.ipls import SigmaPointIpls
from src.smoother.slr.reg_ipls import SigmaPointRegIpls
from src.slr.sigma_points import SigmaPointSlr
from src.sigma_points import SphericalCubature
from src.cost import analytical_smoothing_cost, slr_smoothing_cost
from src.utils import setup_logger
from src.analytics import rmse
from src.analytics import nees
from src.visualization import to_tikz, write_to_tikz_file
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import LmCoordTurn
from data.lm_ieks_paper.coord_turn_example import Type, get_specific_states_from_file, simulate_data
from exp.coord_turn_bearings_only import run_smoothing, calc_iter_metrics, mc_stats


def main():
    args = parse_args()
    log = logging.getLogger(__name__)
    experiment_name = "lm_ieks"
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

    num_iter = 10
    if args.random:
        states, measurements = simulate_data(sens_pos_1, sens_pos_2, std, dt, prior_mean[:-1], time_steps=500)
    else:
        states, measurements, _, xs_ss = get_specific_states_from_file(
            Path.cwd() / "data/lm_ieks_paper", Type.LM, num_iter
        )

    results = []
    cost_fn_eks = partial(
        analytical_smoothing_cost,
        meas=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
        motion_model=motion_model,
        meas_model=meas_model,
    )

    # ms_gn_ieks, Ps_gn_ieks, cost_gn_ieks, rmses_gn_ieks, neeses_gn_ieks = run_smoothing(
    #     Ieks(motion_model, meas_model, num_iter),
    #     states,
    #     measurements,
    #     prior_mean,
    #     prior_cov,
    #     cost_fn_eks,
    #     (np.zeros((measurements.shape[0], prior_mean.shape[0])), None),
    # )
    # results.append(
    #     (ms_gn_ieks, Ps_gn_ieks, cost_gn_ieks[1:], "GN-IEKS"),
    # )
    ms_lm_ieks, Ps_lm_ieks, cost_lm_ieks, rmses_lm_ieks, neeses_lm_ieks = run_smoothing(
        LmIeks(motion_model, meas_model, num_iter, 10, 1e-2, 10),
        states,
        measurements,
        prior_mean,
        prior_cov,
        cost_fn_eks,
        (np.zeros((measurements.shape[0], prior_mean.shape[0])), None),
    )
    assert np.allclose(ms_lm_ieks, xs_ss)
    # err = states - ms_lm_ieks[:, :-1]
    # ind = -1
    # print("err", err[ind, :])
    # err_1 = err[ind, :]
    print("NEES")
    for nees_ in neeses_lm_ieks:
        print(nees_)
    results.append(
        (ms_lm_ieks, Ps_lm_ieks, cost_lm_ieks[1:], "LM-IEKS"),
    )
    # sigma_point_method = SphericalCubature()
    # cost_fn_ipls = partial(
    #     slr_smoothing_cost,
    #     measurements=measurements,
    #     m_1_0=prior_mean,
    #     P_1_0=prior_cov,
    #     motion_model=motion_model,
    #     meas_model=meas_model,
    #     slr=SigmaPointSlr(sigma_point_method),
    # )

    # ms_gn_ipls, Ps_gn_ipls, cost_gn_ipls, rmses_gn_ipls, neeses_gn_ipls = run_smoothing(
    #     SigmaPointIpls(motion_model, meas_model, sigma_point_method, num_iter),
    #     states,
    #     measurements,
    #     prior_mean,
    #     prior_cov,
    #     cost_fn_ipls,
    #     None,
    # )
    # results.append(
    #     (ms_gn_ipls, Ps_gn_ipls, cost_gn_ipls, "GN-IPLS"),
    # )
    # ms_lm_ipls, Ps_lm_ipls, cost_lm_ipls, rmses_lm_ipls, neeses_lm_ipls = run_smoothing(
    #     SigmaPointRegIpls(
    #         motion_model, meas_model, sigma_point_method, num_iter, cost_improv_iter_lim=10, lambda_=1e-2, nu=10
    #     ),
    #     states,
    #     measurements,
    #     prior_mean,
    #     prior_cov,
    #     cost_fn_ipls,
    #     None,
    # )
    # results.append(
    #     (ms_lm_ipls, Ps_lm_ipls, cost_lm_ipls, "LM-IPLS"),
    # )
    plot_results(
        states,
        results,
        None,
    )
    print("GN-NEES")
    for nees_ in neeses_gn_ieks:
        print(nees_)
    plot_results(
        states,
        results,
        None,
    )


# TODO: Sep module?
def plot_results(states, trajs_and_costs, meas):
    means_and_covs = [(ms, Ps, f"{label}-{len(cost)}") for (ms, Ps, cost, label) in trajs_and_costs]
    costs = [
        (cost, f"{label}-{len(cost)}") for (_, _, cost, label) in trajs_and_costs
    ]  # _, (ax_1, ax_2) = plt.subplots(1, 2)
    print("costs:\n", costs)
    fig, ax_1 = plt.subplots()
    vis.plot_2d_est(
        states,
        meas=meas,
        means_and_covs=means_and_covs,
        sigma_level=2,
        skip_cov=50,
        ax=ax_1,
    )
    plt.show()


def plot_cost(ax, costs):
    for (iter_cost, label) in costs:
        ax.semilogy(iter_cost, label=label)
    ax.set_xlabel("Iteration number i")
    ax.set_ylabel("$L_{LM}$")
    ax.set_title("Cost function")


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--random", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main()
