"""Smoother testing script"""
import logging
from functools import partial
import numpy as np
from src.models.coord_turn import CoordTurn
from src.models.range_bearing import RangeBearing
from src.slr.sigma_points import SigmaPointSlr
from src.smoother.slr.ipls import SigmaPointIpls
from src.smoother.ext.ieks import Ieks
from src.sigma_points import SphericalCubature
from src import visualization as vis
from src.utils import setup_logger
from data.coord_turn import get_tricky_data
from src.models.range_bearing import to_cartesian_coords
from src.cost_fn.slr import slr_smoothing_cost


def main():
    log = logging.getLogger(__name__)
    experiment_name = "coord_turn"
    setup_logger(f"logs/{experiment_name}.log", logging.INFO)
    log.info(f"Running experiment: {experiment_name}")
    np.random.seed(2)
    range_ = (0, -1)
    num_iter = 5

    # Motion model
    sampling_period = 0.1
    v_scale = 2
    omega_scale = 2
    sigma_v = v_scale * 1
    sigma_omega = omega_scale * np.pi / 180
    Q = np.diag([0, 0, sampling_period * sigma_v ** 2, 0, sampling_period * sigma_omega ** 2])
    motion_model = CoordTurn(sampling_period, Q)

    # Meas model
    pos = np.array([100, -100])
    sigma_r = 2
    sigma_phi = 0.5 * np.pi / 180

    R = np.diag([sigma_r ** 2, sigma_phi ** 2])
    meas_model = RangeBearing(pos, R)

    # Generate data
    true_states, measurements = get_tricky_data(meas_model, R, range_)
    obs_dims = true_states.shape[1]
    cartes_meas = np.apply_along_axis(partial(to_cartesian_coords, pos=pos), 1, measurements)

    # Prior distr.
    prior_mean = np.array([4.4, 0, 4, 0, 0])
    prior_cov = np.diag([1 ** 2, 1 ** 2, 1 ** 2, (5 * np.pi / 180) ** 2, (1 * np.pi / 180) ** 2])

    cost_fn_ipls = partial(
        slr_smoothing_cost,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
        motion_model=motion_model,
        meas_model=meas_model,
        slr=SigmaPointSlr(SphericalCubature()),
    )

    smoother = SigmaPointIpls(motion_model, meas_model, SphericalCubature(), num_iter)
    mf, Pf, ms, Ps, _ = smoother.filter_and_smooth(measurements, prior_mean, prior_cov, cost_fn_ipls)

    vis.plot_nees_and_2d_est(
        true_states[range_[0] : range_[1], :],
        cartes_meas,
        [
            (mf[:, :obs_dims], Pf[:, :obs_dims, :obs_dims], "filter"),
            (ms[:, :obs_dims], Ps[:, :obs_dims, :obs_dims], "smoother"),
        ],
        sigma_level=3,
        skip_cov=5,
    )


if __name__ == "__main__":
    main()
