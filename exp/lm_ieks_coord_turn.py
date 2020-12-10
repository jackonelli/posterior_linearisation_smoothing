"""Example: Levenberg-Marquardt regularised IEKS smoothing"""
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
from src.models.coord_turn import CoordTurn
from data.lm_coord_turn_example import gen_data, get_specific_states_from_file
from pathlib import Path


def main():
    log = logging.getLogger(__name__)
    experiment_name = "lm_ieks"
    setup_logger(f"logs/{experiment_name}.log", logging.INFO)
    log.info(f"Running experiment: {experiment_name}")
    seed = 2
    np.random.seed(seed)
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
    motion_model = CoordTurn(dt, Q)

    sens_pos_1 = np.array([-1.5, 0.5])
    sens_pos_2 = np.array([1, 1])
    sensors = np.row_stack((sens_pos_1, sens_pos_2))
    std = 0.5
    R = std ** 2 * np.eye(2)
    meas_model = MultiSensorRange(sensors, R)

    prior_mean = np.array([0, 0, 1, 0, 0])
    prior_cov = np.diag([0.1, 0.1, 1, 1, 1])

    # states, measurements = gen_data(sens_pos_1, sens_pos_2, std, dt, prior_mean[:-1], K, seed)

    states, measurements, ss_xf = get_specific_states_from_file(Path.cwd())
    states = states[:K, :]
    measurements = measurements[:K, :]
    ss_xf = ss_xf[:K, :]
    # filter_ = Ekf(motion_model, meas_model)
    # xf, Pf, xp, Pp = filter_.filter_seq(measurements[:K, :], prior_mean, prior_cov)
    eks = Eks(motion_model, meas_model)
    xf, Pf, xs, Ps = eks.filter_and_smooth(measurements, prior_mean, prior_cov)
    num_iter = 5
    ieks = Ieks(motion_model, meas_model, num_iter)
    ixf, iPf, ixs, iPs = ieks.filter_and_smooth(measurements, prior_mean, prior_cov)
    vis.plot_nees_and_2d_est(
        states, measurements, ixf[:, :-1], iPf[:, :-1, :-1], ixs[:, :-1], iPs[:, :-1, :-1], sigma_level=0, skip_cov=20
    )
    # _, ax = plt.subplots()
    # ax.plot(states[:K, 0], states[:K, 1], label="true")
    # ax.plot(ss_xf[:K, 0], ss_xf[:K, 1], label="matlab")
    # ax.plot(xf[:K, 0], xf[:K, 1], label="python")
    # ax.legend()
    # plt.show()


if __name__ == "__main__":
    main()
