"""Example: Levenberg-Marquardt regularised IEKS smoothing

Reproducing the experiment (with two different meas. models) in the paper:

"Levenberg-Marquardt and line-search extended Kalman smoother".

The problem is evaluated for
(Original experiment)
- (GN-)IEKS
- LM-IEKS
(Additional models)
- (GN-)IPLS
- Reg-IPLS (LM-IPLS)
"""

from enum import Enum
from timeit import timeit
from functools import partial
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
from src.smoother.slr.reg_ipls import SigmaPointLmIpls
from src.slr.sigma_points import SigmaPointSlr
from src.sigma_points import SphericalCubature
from src.cost import analytical_smoothing_cost, slr_smoothing_cost, noop_cost, slr_smoothing_cost_pre_comp
from src.utils import setup_logger
from src.analytics import rmse
from src.visualization import to_tikz, write_to_tikz_file
from src.models.range_bearing import MultiSensorRange, MultiSensorBearings
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

    prior_mean = np.array([0, 0, 1, 0, 0])
    prior_cov = np.diag([0.1, 0.1, 1, 1, 1])

    num_iter = 3
    if args.random:
        states, all_meas = simulate_data(sens_pos_1, sens_pos_2, std, dt, prior_mean[:-1], time_steps=500)
    else:
        states, all_meas, _, xs_ss = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.LM, num_iter)

    if args.meas_type == MeasType.Range:
        meas_model = MultiSensorRange(sensors, R)
        measurements = all_meas[:, :2]
    elif args.meas_type == MeasType.Bearings:
        meas_model = MultiSensorBearings(sensors, R)
        measurements = all_meas[:, 2:]

    results = []
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
        slr_smoothing_cost,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
        motion_model=motion_model,
        meas_model=meas_model,
        slr=SigmaPointSlr(sigma_point_method),
    )

    new_cost_fn_ipls = partial(
        slr_smoothing_cost_pre_comp,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
    )
    num_samples = 2
    time_ieks = partial(
        Ieks(motion_model, meas_model, num_iter).filter_and_smooth,
        measurements,
        prior_mean,
        prior_cov,
        partial(noop_cost, covs=None),
    )

    time_lm_ieks = partial(
        LmIeks(motion_model, meas_model, num_iter, 10, 1e-2, 10).filter_and_smooth,
        measurements,
        prior_mean,
        prior_cov,
        cost_fn_eks,
    )
    time_ipls = partial(
        SigmaPointIpls(motion_model, meas_model, sigma_point_method, num_iter).filter_and_smooth,
        measurements,
        prior_mean,
        prior_cov,
        partial(noop_cost, covs=None),
    )
    time_lm_ipls = partial(
        SigmaPointLmIpls(
            motion_model, meas_model, sigma_point_method, num_iter, cost_improv_iter_lim=10, lambda_=1e-2, nu=10
        ).filter_and_smooth,
        measurements,
        prior_mean,
        prior_cov,
        new_cost_fn_ipls,
    )
    time_ieks = timeit(time_ieks, number=num_samples) / (num_iter * num_samples)
    time_lm_ieks = timeit(time_lm_ieks, number=num_samples) / (num_iter * num_samples)
    time_ipls = timeit(time_ipls, number=num_samples) / (num_iter * num_samples)
    time_lm_ipls = timeit(time_lm_ipls, number=num_samples) / (num_iter * num_samples)
    print(f"IEKS: {time_ieks:.2f} s, 100.0%")
    print(f"LM-IEKS: {time_lm_ieks:.2f} s, {time_lm_ieks/time_ieks*100:.2f}%")
    print(f"IPLS: {time_ipls:.2f} s, {time_ipls/time_ieks*100:.2f}%")
    print(f"LM-IPLS: {time_lm_ipls:.2f} s, {time_lm_ipls/time_ieks*100:.2f}%")


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
    parser.add_argument("--meas_type", type=MeasType, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    main()
