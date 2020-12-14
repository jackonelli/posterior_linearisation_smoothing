"""Example: Levenberg-Marquardt regularised IEKS smoothing

Reproducing the experiment in the paper:

"Levenberg-marquardt and line-search extended kalman smoother"
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from src import visualization as vis
from src.filter.ekf import Ekf
from src.smoother.eks import Eks
from src.smoother.ieks import Ieks
from src.utils import setup_logger
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import LmCoordTurn
from data.lm_coord_turn_example import gen_data, get_specific_states_from_file
from pathlib import Path


def main():
    log = logging.getLogger(__name__)
    experiment_name = "lm_ieks"
    setup_logger(f"logs/{experiment_name}.log", logging.INFO)
    log.info(f"Running experiment: {experiment_name}")
    seed = 2
    np.random.seed(seed)
    num_iter = 1
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

    # states, measurements = gen_data(sens_pos_1, sens_pos_2, std, dt, prior_mean[:-1], K, seed)

    states, measurements, ss_xf, ss_xs = get_specific_states_from_file(Path.cwd())
    states = states[:K, :]
    measurements = measurements[:K, :]
    ss_xf = ss_xf[:K, :]
    # filter_ = Ekf(motion_model, meas_model)
    # xf, Pf, xp, Pp = filter_.filter_seq(measurements[:K, :], prior_mean, prior_cov)
    eks = Ieks(motion_model, meas_model, num_iter)
    xf, Pf, xs, Ps = eks.filter_and_smooth(measurements, prior_mean, prior_cov)
    vis.plot_nees_and_2d_est(
        true_x=states,
        meas=None,
        xf=None,
        Pf=None,
        xs=xs[:, :-1],
        Ps=Ps[:, :-1, :-1],
        sigma_level=2,
        skip_cov=50,
    )


if __name__ == "__main__":
    main()
