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
from src.smoother.ext.eks import Eks
from src.smoother.ext.ieks import Ieks
from src.smoother.ext.lm_ieks import LmIeks
from src.smoother.ipls import Ipls
from src.utils import setup_logger
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import LmCoordTurn
from data.lm_ieks_paper.coord_turn_example import Type, get_specific_states_from_file
from exp.matlab_comp import gn_ieks, basic_eks

from src.smoother.ext.cost import cost


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

    states, measurements, ss_mf, ss_ms = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.LM)
    states = states[:K, :]
    measurements = measurements[:K, :]
    num_iter = 10
    smoother = Ieks(motion_model, meas_model, num_iter)
    # lambda_ = 1e-2
    # nu = 10
    # smoother = LmIeks(motion_model, meas_model, num_iter, lambda_, nu)
    # mf, Pf, ms, Ps = smoother.filter_and_smooth(measurements, prior_mean, prior_cov)
    mf, Pf, ms, Ps = smoother.filter_and_smooth_with_init_traj(
        measurements, prior_mean, prior_cov, np.zeros((K, prior_mean.shape[0])), 1
    )
    # ss_mf, ss_Pf, ss_ms, ss_Ps = gn_eks(
    #     measurements,
    #     prior_mean,
    #     prior_cov,
    #     Q,
    #     R,
    #     motion_model.mapping,
    #     motion_model.jacobian,
    #     meas_model.mapping,
    #     meas_model.jacobian,
    #     num_iter,
    #     np.zeros((K, prior_mean.shape[0])),
    # )

    # assert np.allclose(ms, ss_ms)
    # print("Ieks: ", cost(ms, measurements, prior_mean, prior_cov, motion_model.mapping, meas_model.mapping, Q, R))
    # print("Matl: ", cost(ss_ms, measurements, prior_mean, prior_cov, motion_model.mapping, meas_model.mapping, Q, R))
    # print("Comp: ", acost(ss_ms, measurements, prior_mean, prior_cov, motion_model, meas_model))
    # vis.cmp_states(ms, ss_ms)
    vis.plot_2d_est(
        true_x=states,
        meas=None,
        means_and_covs=[(ms, Ps, f"ms_{num_iter}"), (ss_ms, ss_Ps, f"ss_ms_{num_iter}")],
        sigma_level=2,
        skip_cov=50,
    )


if __name__ == "__main__":
    main()
