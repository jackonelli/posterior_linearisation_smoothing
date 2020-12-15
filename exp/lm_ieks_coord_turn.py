"""Example: Levenberg-Marquardt regularised IEKS smoothing

Reproducing the experiment in the paper:

"Levenberg-marquardt and line-search extended kalman smoother"
"""
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src import visualization as vis
from src.filter.ekf import Ekf
from src.smoother.eks import Eks
from src.smoother.ieks import Ieks
from src.smoother.ipls import Ipls
from src.utils import setup_logger
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import LmCoordTurn
from data.lm_ieks_paper.coord_turn_example import Type, get_specific_states_from_file
from exp.matlab_comp import gn_eks


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
    motion_model = LmCoordTurn(dt, Q)

    sens_pos_1 = np.array([-1.5, 0.5])
    sens_pos_2 = np.array([1, 1])
    sensors = np.row_stack((sens_pos_1, sens_pos_2))
    std = 0.5
    R = std ** 2 * np.eye(2)
    meas_model = MultiSensorRange(sensors, R)

    prior_mean = np.array([0, 0, 1, 0, 0])
    prior_cov = np.diag([0.1, 0.1, 1, 1, 1])

    states, measurements, ss_xf, ss_xs = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.GN)
    states = states[:K, :]
    measurements = measurements[:K, :]
    num_iter = 10
    smoother = Ipls(motion_model, meas_model, num_iter)
    xf, Pf, xs, Ps = smoother.filter_and_smooth(measurements, prior_mean, prior_cov)

    vis.plot_2d_est(
        true_x=states,
        meas=None,
        xf=xs[:, :-1],
        Pf=Ps[:, :-1, :-1],
        xs=ss_xs[:, :-1],
        Ps=Ps[:, :-1, :-1],
        sigma_level=0,
        skip_cov=50,
    )


if __name__ == "__main__":
    main()
