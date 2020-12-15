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

    # states, measurements = gen_data(sens_pos_1, sens_pos_2, std, dt, prior_mean[:-1], K, seed)

    states, measurements, _, _ = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.GN)
    states = states[:K, :]
    measurements = measurements[:K, :]
    # ss_xf = ss_xf[:K, :]
    # filter_ = Ekf(motion_model, meas_model)
    # xf, Pf, xp, Pp = filter_.filter_seq(measurements[:K, :], prior_mean, prior_cov)
    num_iter = 10
    eks = Ieks(motion_model, meas_model, num_iter)
    xf, Pf, xs, Ps = eks.filter_and_smooth(measurements, prior_mean, prior_cov)
    ss_xf, ss_Pf, ss_xs, ss_Ps = gn_eks(
        measurements,
        prior_mean,
        prior_cov,
        Q,
        R,
        motion_model.mapping,
        motion_model.jacobian,
        meas_model.mapping,
        meas_model.jacobian,
        num_iter,
        np.zeros((K, prior_mean.shape[0])),
    )

    # assert np.allclose(eks._current_means, ss_xs)
    # assert np.allclose(xf, ss_xf)
    # assert np.allclose(xs, ss_xs)
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
