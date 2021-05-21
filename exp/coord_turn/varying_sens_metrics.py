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
from src.smoother.slr.lm_ipls import SigmaPointLmIpls
from src.slr.sigma_points import SigmaPointSlr
from src.sigma_points import SphericalCubature
from src.cost import analytical_smoothing_cost_time_dep, slr_smoothing_cost, slr_smoothing_cost_pre_comp, slr_noop_cost
from src.utils import setup_logger
from src.analytics import rmse
from src.visualization import to_tikz, write_to_tikz_file
from src.utils import tikz_2d_tab_format
from src.models.range_bearing import BearingsVaryingSensors, MultiSensorBearings
from src.models.coord_turn import CoordTurn
from data.lm_ieks_paper.coord_turn_example import Type, get_specific_states_from_file, simulate_data
from exp.ct_bearings_only import run_smoothing, calc_iter_metrics, mc_stats


def main():
    args = parse_args()
    log = logging.getLogger(__name__)
    experiment_name = "ct_only_varying_sens"
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
    single_meas_time_steps = set(list(range(0, 500, 50))[1:])
    meas_model = BearingsVaryingSensors(double_meas_model, single_meas_model, single_meas_time_steps)

    prior_mean = np.array([0, 0, 1, 0, 0])
    prior_cov = np.diag([0.1, 0.1, 1, 1, 1])

    num_iter = args.num_iter

    if args.random:
        states, measurements = simulate_data(motion_model, double_meas_model, prior_mean[:-1], time_steps=500)
    else:
        states, all_meas, _, xs_ss = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.LM, num_iter)
        measurements = all_meas[:, 2:]

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

    ms_gn_ieks, Ps_gn_ieks, cost_gn_ieks, rmses_gn_ieks, neeses_gn_ieks = run_smoothing(
        Ieks(motion_model, meas_model, num_iter),
        states,
        measurements,
        prior_mean,
        prior_cov,
        cost_fn_eks,
        (np.zeros((len(measurements), prior_mean.shape[0])), None),
    )
    results.append(
        (ms_gn_ieks, Ps_gn_ieks, cost_gn_ieks[1:], "GN-IEKS"),
    )

    ms_lm_ieks, Ps_lm_ieks, cost_lm_ieks, rmses_lm_ieks, neeses_lm_ieks = run_smoothing(
        LmIeks(motion_model, meas_model, num_iter, 10, 1e1, 10),
        states,
        measurements,
        prior_mean,
        prior_cov,
        cost_fn_eks,
        (np.zeros((len(measurements), prior_mean.shape[0])), None),
    )
    results.append(
        (ms_lm_ieks, Ps_lm_ieks, cost_lm_ieks[1:], "LM-IEKS"),
    )

    sigma_point_method = SphericalCubature()
    cost_fn_ipls = partial(
        slr_smoothing_cost_pre_comp,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
    )

    ms_gn_ipls, Ps_gn_ipls, cost_gn_ipls, rmses_gn_ipls, neeses_gn_ipls = run_smoothing(
        SigmaPointIpls(motion_model, meas_model, sigma_point_method, num_iter),
        states,
        measurements,
        prior_mean,
        prior_cov,
        slr_noop_cost,
        None,
    )
    results.append(
        (ms_gn_ipls, Ps_gn_ipls, cost_gn_ipls, "GN-IPLS"),
    )
    ms_lm_ipls, Ps_lm_ipls, cost_lm_ipls, rmses_lm_ipls, neeses_lm_ipls = run_smoothing(
        SigmaPointLmIpls(
            motion_model, meas_model, sigma_point_method, num_iter, cost_improv_iter_lim=10, lambda_=1e0, nu=10
        ),
        states,
        measurements,
        prior_mean,
        prior_cov,
        cost_fn_ipls,
        None,
    )
    results.append(
        (ms_lm_ipls, Ps_lm_ipls, cost_lm_ipls, "LM-IPLS"),
    )
    plot_results(
        states,
        results,
        states[list(single_meas_time_steps)],
    )
    gen_tikz(states, results)


def gen_tikz(states, results):
    dir_ = Path.cwd() / "tmp_results"
    for ms, _, _, label in results:
        write_to_tikz_file(tikz_2d_tab_format(ms[:, 0], ms[:, 1]), dir_, f"{label.lower()}.data")


def modify_meas(measurements, states, meas_model, resample):
    if resample:
        measurements = meas_model._default_model.sample(states)
    # np array to list of np array
    measurements = [meas for meas in measurements]
    change_tss = meas_model.alt_tss()
    new_meas = meas_model._alt_model.sample(states[change_tss - 1, :])
    for counter, ts in enumerate(change_tss):
        measurements[ts - 1] = new_meas[counter]
    return measurements


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

    return parser.parse_args()


if __name__ == "__main__":
    main()
