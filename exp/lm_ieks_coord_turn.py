"""Example: Levenberg-Marquardt regularised IEKS smoothing

Reproducing the experiment in the paper:

"Levenberg-Marquardt and line-search extended Kalman smoother"
"""

import logging
from pathlib import Path
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from src import visualization as vis
from src.filter.ekf import Ekf
from src.smoother.ext.eks import Eks
from src.smoother.ext.ieks import Ieks
from src.smoother.ext.lm_ieks import LmIeks
from src.smoother.ext.cost import cost
from src.utils import setup_logger
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import LmCoordTurn
from data.lm_ieks_paper.coord_turn_example import Type, get_specific_states_from_file

from src.smoother.ext.cost import cost


def main():
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    log = logging.getLogger(__name__)
    experiment_name = "lm_ieks"
    setup_logger(f"logs/{experiment_name}.log", logging.DEBUG)
    log.info(f"Running experiment: {experiment_name}")
    K = 500

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
    states, measurements, _, _ = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.LM, num_iter)

    cost_fn = partial(
        cost,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
        motion_model=motion_model,
        meas_model=meas_model,
    )
    ms_gn, Ps_gn, cost_gn = gn_ieks(motion_model, meas_model, num_iter, measurements, prior_mean, prior_cov, cost_fn)
    ms_lm, Ps_lm, cost_lm = lm_ieks(motion_model, meas_model, num_iter, measurements, prior_mean, prior_cov, cost_fn)
    plot_results(states, [(ms_gn, Ps_gn, cost_gn[1:], "GN-IEKS"), (ms_lm, Ps_lm, cost_lm[1:], "LM-IEKS")])


def gn_ieks(motion_model, meas_model, num_iter, measurements, prior_mean, prior_cov, cost_fn):
    K = measurements.shape[0]
    # No initial covariances nec.
    init_traj = (np.zeros((K, prior_mean.shape[0])), None)
    smoother = Ieks(motion_model, meas_model, num_iter)
    # Note that the paper uses m_k = 0, k = 1, ..., K as the initial trajectory
    # This is the reason for not using the ordinary `filter_and_smooth` method.
    _, _, ms, Ps, iter_cost = smoother.filter_and_smooth_with_init_traj(
        measurements, prior_mean, prior_cov, init_traj, 1, cost_fn
    )
    return ms, Ps, iter_cost


def lm_ieks(motion_model, meas_model, num_iter, measurements, prior_mean, prior_cov, cost_fn):
    lambda_ = 1e-2
    nu = 10
    K = measurements.shape[0]
    init_traj = (np.zeros((K, prior_mean.shape[0])), None)
    smoother = LmIeks(motion_model, meas_model, num_iter, lambda_, nu)
    # Note that the paper uses m_k = 0, k = 1, ..., K as the initial trajectory
    # This is the reason for not using the ordinary `filter_and_smooth` method.
    _, _, ms, Ps, iter_cost = smoother.filter_and_smooth_with_init_traj(
        measurements, prior_mean, prior_cov, init_traj, 1, cost_fn
    )
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
