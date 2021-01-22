"""Example: Levenberg-Marquardt regularised IEKS smoothing

Reproducing the experiment in the paper:

"Levenberg-marquardt and line-search extended Kalman smoother"
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
from src.smoother.ipls import Ipls
from src.utils import setup_logger
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import LmCoordTurn
from data.lm_ieks_paper.coord_turn_example import Type, get_specific_states_from_file

from src.smoother.ext.cost import cost


def main():
    log = logging.getLogger(__name__)
    experiment_name = "lm_ieks"
    setup_logger(f"logs/{experiment_name}.log", logging.INFO)
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

    lambda_ = 1e-2
    nu = 10
    cost_fn = partial(
        cost,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
        motion_model=motion_model,
        meas_model=meas_model,
    )
    smoother = LmIeks(motion_model, meas_model, num_iter, lambda_, nu)
    # Note that the paper uses m_k = 0, k = 1, ..., K as the initial trajectory
    # This is the reason for not using the ordinary `filter_and_smooth` method.
    mf, Pf, ms, Ps, iter_cost = smoother.filter_and_smooth_with_init_traj(
        measurements, prior_mean, prior_cov, np.zeros((K, prior_mean.shape[0])), 1, cost_fn
    )

    plot_results(states, ms, Ps, iter_cost[1:])


def plot_results(states, ms, Ps, iter_cost):
    _, (ax_1, ax_2) = plt.subplots(1, 2)
    plot_smooth_traj(ax_1, states, ms, Ps, f"LM-IEKS-{len(iter_cost)}")
    plot_cost(ax_2, iter_cost[1:])
    plt.show()


def plot_cost(ax, iter_cost):
    ax.plot(iter_cost)
    ax.set_xlabel("Iteration number i")
    ax.set_ylabel("$L_{LM}$")
    ax.set_title("Cost function")


def plot_smooth_traj(ax, true_x, ms, Ps, label):
    vis.plot_2d_est(
        true_x,
        meas=None,
        means_and_covs=[
            (ms, Ps, label),
        ],
        sigma_level=2,
        skip_cov=50,
        ax=ax,
    )


if __name__ == "__main__":
    main()
