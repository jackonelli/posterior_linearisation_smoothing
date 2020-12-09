"""Example: Levenberg-Marquardt regularised IEKS smoothing"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from src import visualization as vis
from src.filter.kalman import KalmanFilter
from src.smoother.rts import RtsSmoother
from src.utils import setup_logger
from src.models.range_bearing import MultiSensorRange
from data.affine import sim_affine_state_seq, sim_affine_meas_seq


def main():
    log = logging.getLogger(__name__)
    experiment_name = "lm_ieks"
    setup_logger(f"logs/{experiment_name}.log", logging.INFO)
    log.info(f"Running experiment: {experiment_name}")
    K = 20
    sens_pos_1 = (np.array([-1.5, 0.5]),)
    sens_pos_2 = np.array([1, 1])
    sensors = np.row_stack((sens_pos_1, sens_pos_2))
    std = 0.5
    R = std ** 2 * np.eye(2)
    meas_model = MultiSensorRange(sensors, R)
    # data = gen_data(
    #     sens_pos_1=np.array([-1.5, 0.5]),
    #     sens_pos_2=np.array([1, 1]),  # Position of sensor 2
    #     std=0.5,  # Standard deviation of measurements
    #     dt=0.01,  # Sampling period
    #     x_0=np.array([0.1, 0.2, 1, 0]),  # Initial state
    #     time_steps=500,
    #     seed=4,
    # )

    prior_mean = np.array([0, 0, 1, 0, 0])
    prior_cov = np.diag([0.1, 0.1, 1, 1, 1])
    y_mean = meas_model.mapping(prior_mean)
    jac = meas_model.jacobian(prior_mean)
    print(jac)
    # T = 1
    # A = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]])
    # b = 0 * np.ones((4,))
    # Q = np.array(
    #     [
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 0, 1.5, 0],
    #         [0, 0, 0, 1.5],
    #     ]
    # )
    # H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    # c = np.zeros((H @ prior_mean).shape)
    # R = 2 * np.eye(2)

    # true_x = sim_affine_state_seq(prior_mean, prior_cov, motion_model, K)
    # y = sim_affine_meas_seq(true_x, H, R)

    # analytical_smooth = RtsSmoother(motion_model, meas_model)
    # xf, Pf, xs, Ps = analytical_smooth.filter_and_smooth(y, prior_mean, prior_cov)
    # vis.plot_nees_and_2d_est(true_x, y, xf, Pf, xs, Ps, sigma_level=3, skip_cov=2)


if __name__ == "__main__":
    main()
