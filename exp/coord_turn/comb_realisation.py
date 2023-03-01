"""Example: Levenberg-Marquardt regularised IEKS smoothing

Reproducing the experiment (with two different meas. models) in the paper:

"Levenberg-Marquardt and line-search extended Kalman smoother".

The problem is evaluated for
(Original experiment)
- IEKS
- LM-IEKS
(Additional models)
- IPLS
- Reg-IPLS (LM-IPLS)
"""

import logging
import argparse
from pathlib import Path
from functools import partial
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from src.smoother.ext.ieks import Ieks
from src.smoother.ext.lm_ieks import LmIeks
from src.smoother.ext.ls_ieks import LsIeks
from src.smoother.slr.ipls import SigmaPointIpls
from src.smoother.slr.ls_ipls import SigmaPointLsIpls
from src.smoother.slr.lm_ipls import SigmaPointLmIpls
from src.line_search import ArmijoWolfeLineSearch
from src.slr.sigma_points import SigmaPointSlr
from src.sigma_points import SphericalCubature
from src.cost_fn.ext import analytical_smoothing_cost_time_dep, dir_der_analytical_smoothing_cost
from src.cost_fn.slr import slr_smoothing_cost_pre_comp, slr_smoothing_cost_means
from src.utils import setup_logger, tikz_2d_traj
from src.visualization import to_tikz, write_to_tikz_file
from src.models.range_bearing import MultiSensorRange, MultiSensorBearings, BearingsVaryingSensors
from src.models.coord_turn import CoordTurn
from data.lm_ieks_paper.coord_turn_example import Type, get_specific_states_from_file, simulate_data
from exp.coord_turn.common import MeasType, run_smoothing, plot_results, modify_meas


def main():
    args = parse_args()
    log = logging.getLogger(__name__)
    experiment_name = "ct_experiment_realisation"
    setup_logger(f"logs/{experiment_name}.log", logging.INFO)
    log.info(f"Running experiment: {experiment_name}")
    if not args.random:
        np.random.seed(2)

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

    prior_mean = np.array([0, 0, 1, 0, 0])
    prior_cov = np.diag([0.1, 0.1, 1, 1, 1])

    # LM parameters
    lambda_ = 1e-0
    nu = 10
    # Armijo-Wolfe parameters
    c_1, c_2 = 0.1, 0.9

    num_iter = args.num_iter
    if args.meas_type == MeasType.Range:
        meas_model = MultiSensorRange(sensors, R)
        meas_cols = np.array([0, 1])
    elif args.meas_type == MeasType.Bearings:
        meas_model = MultiSensorBearings(sensors, R)
        meas_cols = np.array([2, 3])
    else:
        log.error("Invalid meas_type arg, expected ('range' | 'bearing')")
        return

    log.info("Generating states and measurements.")
    if args.random:
        states, measurements = simulate_data(motion_model, meas_model, prior_mean[:-1], time_steps=500)
    else:
        states, all_meas, _, xs_ss = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.LM, num_iter)
        measurements = all_meas[:, meas_cols]

    if args.var_sensors:
        single_sensor = sens_pos_2.reshape((1, 2))
        std = 0.001
        R_certain = std ** 2 * np.eye(1)
        single_meas_model = MultiSensorBearings(single_sensor, R_certain)
        # Create a set of time steps where the original two sensor measurements are replaced with single ones.
        # single_meas_time_steps = set(list(range(0, 100, 5))[1:])
        single_meas_time_steps = set(list(range(0, 500, 50))[1:])
        meas_model = BearingsVaryingSensors(meas_model, single_meas_model, single_meas_time_steps)
        # Change measurments so that some come from the alternative model.
        measurements = modify_meas(measurements, states, meas_model, True)

    results = []
    cost_fn_eks = partial(
        analytical_smoothing_cost_time_dep,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
        motion_model=motion_model,
        meas_model=meas_model,
    )

    D_x = prior_mean.shape[0]
    K = len(measurements)
    init_traj = (np.zeros((K, D_x)), np.array(K * [prior_cov]))

    log.info("Running IEKS...")
    ms_ieks, Ps_ieks, cost_ieks, _, _ = run_smoothing(
        Ieks(motion_model, meas_model, num_iter), states, measurements, prior_mean, prior_cov, cost_fn_eks, init_traj
    )
    results.append(
        (ms_ieks, Ps_ieks, cost_ieks[1:], "IEKS"),
    )

    sigma_point_method = SphericalCubature()

    cost_fn_ipls = partial(
        slr_smoothing_cost_pre_comp, measurements=measurements, m_1_0=prior_mean, P_1_0_inv=np.linalg.inv(prior_cov)
    )

    # log.info("Running IPLS...")
    # ms_ipls, Ps_ipls, cost_ipls, _, _ = run_smoothing(
    #     SigmaPointIpls(motion_model, meas_model, sigma_point_method, num_iter),
    #     states,
    #     measurements,
    #     prior_mean,
    #     prior_cov,
    #     None,
    #     init_traj,
    # )
    # results.append(
    #     (ms_ipls, Ps_ipls, cost_ipls, "IPLS"),
    # )

    log.info("Running LM-IPLS...")
    ms_lm_ipls, Ps_lm_ipls, cost_lm_ipls, _, _ = run_smoothing(
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
    results.append(
        (ms_lm_ipls, Ps_lm_ipls, cost_lm_ipls, "LM-IPLS"),
    )

    log.info("Running LM-IPLS + IEKS...")
    lm_ipls_traj = (deepcopy(ms_lm_ipls), deepcopy(Ps_lm_ipls))
    ms_comb, Ps_comb, cost_comb, rmse_comb, nees_comb = run_smoothing(
        Ieks(motion_model, meas_model, num_iter),
        states,
        measurements,
        prior_mean,
        prior_cov,
        cost_fn_eks,
        lm_ipls_traj,
    )
    results.append(
        (ms_comb, Ps_comb, cost_comb, "LM-IPLS + IEKS"),
    )

    for ms, _, _, label in results:
        tikz_2d_traj(Path.cwd() / "tikz", ms[:, :2], label)
    plot_results(
        states,
        results,
        None,
    )


def plot_cost(ax, costs):
    for (iter_cost, label) in costs:
        ax.semilogy(iter_cost, label=label)
    ax.set_xlabel("Iteration number i")
    ax.set_ylabel("$L_{LM}$")
    ax.set_title("Cost function")


def parse_args():
    parser = argparse.ArgumentParser(description="LM-IEKS paper experiment.")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--var_sensors", action="store_true")
    parser.add_argument("--meas_type", type=MeasType, required=True)
    parser.add_argument("--num_iter", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    main()
