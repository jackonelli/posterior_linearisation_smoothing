"""Smoother testing script"""
import logging
from functools import partial
import numpy as np
from scipy.stats import multivariate_normal as mvn
from src.models.range_bearing import to_cartesian_coords
from src.models.coord_turn import CoordTurn
from src.models.range_bearing import RangeBearing
from src.filter.slr import SigmaPointSlrFilter

# from src.smoother.slr import SigmaPointSlrSmoother
from src.smoother.ipls import Ipls
from src import visualization as vis
from src.utils import setup_logger
from data.coord_turn import get_tricky_data


def main():
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    log = logging.getLogger(__name__)
    experiment_name = "coord_turn"
    setup_logger(f"logs/{experiment_name}.log", logging.INFO)
    log.info(f"Running experiment: {experiment_name}")
    # np.random.seed(1)
    range_ = (0, 200)
    num_iter = 10

    # Motion model
    sampling_period = 0.1
    v_scale = 1
    omega_scale = 1
    sigma_v = v_scale * 1
    sigma_omega = omega_scale * np.pi / 180
    Q = np.diag([0, 0, sampling_period * sigma_v ** 2, 0, sampling_period * sigma_omega ** 2])
    motion_model = CoordTurn(sampling_period, Q)

    # Meas model
    pos = np.array([100, -100])
    sigma_r = 1
    sigma_phi = 0.5 * np.pi / 180

    R = np.diag([sigma_r ** 2, sigma_phi ** 2])
    meas_model = RangeBearing(pos, R)

    # Generate data
    true_states, measurements = get_tricky_data(meas_model, R, range_)
    obs_dims = true_states.shape[1]
    cartes_meas = np.apply_along_axis(partial(to_cartesian_coords, pos=pos), 1, measurements)

    # Prior distr.
    x_0 = np.array([4.4, 0, 4, 0, 0])
    P_0 = np.diag([1 ** 2, 1 ** 2, 1 ** 2, (5 * np.pi / 180) ** 2, (1 * np.pi / 180) ** 2])

    smoother = Ipls(motion_model, meas_model, num_iter)
    xf, Pf, xs, Ps = smoother.filter_and_smooth(measurements, x_0, P_0)

    vis.plot_nees_and_2d_est(
        true_states[range_[0] : range_[1], :],
        cartes_meas,
        xf[:, :obs_dims],
        Pf[:, :obs_dims, :obs_dims],
        xs[:, :obs_dims],
        Ps[:, :obs_dims, :obs_dims],
        sigma_level=3,
        skip_cov=5,
    )


if __name__ == "__main__":
    main()
