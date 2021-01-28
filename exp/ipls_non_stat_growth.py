"""Example: Non-stationary growth model

Reproducing the experiment in the paper:

    "Iteraterd posterior linearization smoother"
"""

import logging
from pathlib import Path
import numpy as np
from src import visualization as vis
from src.smoother.slr.ipls import SigmaPointIpls
from src.utils import setup_logger
from src.models.nonstationary_growth import NonStationaryGrowth
from src.models.cubic import Cubic
from src.sigma_points import UnscentedTransform
from src.filter.iplf import SigmaPointIplf
from src.filter.prlf import SigmaPointPrLf
from data.ipls_paper.data import get_specific_states_from_file, gen_measurements


def main():
    log = logging.getLogger(__name__)
    experiment_name = "ipls"
    setup_logger(f"logs/{experiment_name}.log", logging.INFO)
    log.info(f"Running experiment: {experiment_name}")
    K = 50
    D_x = 1
    num_mc_runs = 1000  # 1000 in original exp
    num_mc_per_traj = 50
    num_trajs = num_mc_runs // num_mc_per_traj
    trajs, noise = get_specific_states_from_file(Path.cwd() / "data/ipls_paper")
    assert trajs.shape == (K, num_trajs)
    assert noise.shape == (K, num_mc_runs)

    motion_model = NonStationaryGrowth(alpha=0.9, beta=10, gamma=8, delta=1.2, proc_noise=1)
    meas_model = Cubic(coeff=1/20, proc_noise=1)

    # The paper simply states that "[t]he SLRs have been implemented using the unscented transform
    # with N s = 2n x + 1 sigma-points and the weight of the sigma-point located on the mean is 1/3."

    # The following settings ensures that:
    # a) w_0 (the weight of the mean sigma point) is 1/3
    # b) there is no difference for the weights for the mean and cov estimation
    sigma_point_method = UnscentedTransform(1, 0, 1 / 2)
    assert sigma_point_method.weights(D_x)[0][0] == 1 / 3
    assert np.allclose(sigma_point_method.weights(D_x)[0], sigma_point_method.weights(D_x)[1])

    prior_mean = np.atleast_1d(5)
    prior_cov = np.atleast_2d([4])

    prlf = SigmaPointPrLf(motion_model, meas_model, sigma_point_method)
    for mc_iter in range(num_mc_runs):
        # initialise storage
        traj_idx = _mc_iter_to_traj_idx(mc_iter, num_mc_per_traj)
        traj = trajs[:, traj_idx]
        meas = gen_measurements(traj, noise[:, mc_iter])
        prlf.filter_seq(meas, prior_mean, prior_cov)


def _mc_iter_to_traj_idx(mc_iter: int, num_mc_per_traj) -> int:
    return int(mc_iter / num_mc_per_traj)


if __name__ == "__main__":
    main()
