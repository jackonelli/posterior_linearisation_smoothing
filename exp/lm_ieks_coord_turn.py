"""Example: Levenberg-Marquardt regularised IEKS smoothing"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from src import visualization as vis
from src.filter.ekf import Ekf
from src.smoother.rts import RtsSmoother
from src.utils import setup_logger
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import CoordTurn
from data.lm_coord_turn_example import get_specific_states_from_file
from pathlib import Path


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
    motion_model = CoordTurn(dt, Q)

    sens_pos_1 = (np.array([-1.5, 0.5]),)
    sens_pos_2 = np.array([1, 1])
    sensors = np.row_stack((sens_pos_1, sens_pos_2))
    std = 0.5
    R = std ** 2 * np.eye(2)
    meas_model = MultiSensorRange(sensors, R)
    states, measurements = get_specific_states_from_file(Path.cwd())

    prior_mean = np.array([0, 0, 1, 0, 0])
    prior_cov = np.diag([0.1, 0.1, 1, 1, 1])

    filter_ = Ekf(motion_model, meas_model)
    xf, Pf, xp, Pp = filter_.filter_seq(measurements, prior_mean, prior_cov)
    # xf, Pf, xs, Ps = analytical_smooth.filter_and_smooth(y, prior_mean, prior_cov)
    vis.plot_nees_and_2d_est(states, measurements, xf[:, :-1], Pf[:, :-1, :-1], None, None, sigma_level=0, skip_cov=20)


if __name__ == "__main__":
    main()
