"""Example: Non-stationary growth model

Reproducing the experiment in the paper:

    "Iteraterd posterior linearization smoother"
"""

import argparse
from enum import Enum
import logging
from functools import partial
import numpy as np
from src import visualization as vis
from src.utils import setup_logger
from src.models.nonstationary_growth import NonStationaryGrowth
from src.models.cubic import Cubic
from src.models.quadratic import Quadratic
from src.sigma_points import UnscentedTransform
from src.smoother.ext.ieks import Ieks
from src.smoother.ext.lm_ieks import LmIeks
from src.smoother.slr.ipls import SigmaPointIpls
from src.smoother.slr.lm_ipls import SigmaPointLmIpls
from data.ipls_paper.data import simulate_data
from src.cost_fn.slr import slr_smoothing_cost_pre_comp
from src.cost_fn.ext import analytical_smoothing_cost_time_dep
import matplotlib.pyplot as plt


def main():
    np.random.seed(1)
    args = parse_args()
    log = logging.getLogger(__name__)
    experiment_name = "ipls_non_stat_growth"
    setup_logger(f"logs/{experiment_name}.log", logging.DEBUG)
    log.info(f"Running experiment: {experiment_name}")
    K = 50
    D_x = 1
    motion_model = NonStationaryGrowth(alpha=0.9, beta=10, gamma=8, delta=1.2, proc_noise=1)

    meas_model = (
        Cubic(coeff=1 / 20, meas_noise=1) if args.meas_type == MeasType.Cubic else Quadratic(coeff=1 / 20, meas_noise=1)
    )

    # LM params
    lambda_ = 1e-2
    nu = 10

    prior_mean = np.atleast_1d(5)
    prior_cov = np.atleast_2d([4])

    states, meas = simulate_data(K, prior_mean, prior_cov, motion_model, meas_model)

    # The paper simply states that "[t]he SLRs have been implemented using the unscented transform
    # with N = 2 D_x + 1 sigma-points and the weight of the sigma-point located on the mean is 1/3."

    # The following settings ensures that:
    # a) w_0 (the weight of the mean sigma point) is 1/3
    # b) there is no difference for the weights for the mean and cov estimation
    sigma_point_method = UnscentedTransform(1, 0, 1 / 2)
    assert sigma_point_method.weights(D_x)[0][0] == 1 / 3
    assert np.allclose(sigma_point_method.weights(D_x)[0], sigma_point_method.weights(D_x)[1])

    results = []
    cost_fn_eks = partial(
        analytical_smoothing_cost_time_dep,
        measurements=meas,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
        motion_model=motion_model,
        meas_model=meas_model,
    )

    ieks = Ieks(motion_model, meas_model, args.num_iter)
    ms_ieks, Ps_ieks, cost_ieks, rmses_ieks, neeses_ieks = ieks.filter_and_smooth(
        meas,
        prior_mean,
        prior_cov,
        cost_fn_eks,
    )
    results.append(
        (ms_ieks, Ps_ieks, "IEKS"),
    )
    lm_ieks = LmIeks(motion_model, meas_model, args.num_iter, 10, lambda_=lambda_, nu=nu)
    ms_lm_ieks, Ps_lm_ieks, cost_lm_ieks, rmses_lm_ieks, neeses_lm_ieks = lm_ieks.filter_and_smooth(
        meas,
        prior_mean,
        prior_cov,
        cost_fn_eks,
    )
    results.append(
        (ms_lm_ieks, Ps_lm_ieks, "LM-IEKS"),
    )
    cost_fn_ipls = partial(
        slr_smoothing_cost_pre_comp,
        measurements=meas,
        m_1_0=prior_mean,
        P_1_0_inv=np.linalg.inv(prior_cov),
    )

    ipls = SigmaPointIpls(motion_model, meas_model, sigma_point_method, args.num_iter)
    _, _, ipls_ms, ipls_Ps, _ = ipls.filter_and_smooth(meas, prior_mean, prior_cov, cost_fn=cost_fn_ipls)

    results.append((ipls_ms, ipls_Ps, "IPLS"))
    lm_ipls = SigmaPointLmIpls(
        motion_model, meas_model, sigma_point_method, args.num_iter, cost_improv_iter_lim=10, lambda_=lambda_, nu=nu
    )

    _, _, lm_ipls_ms, lm_ipls_Ps, _ = lm_ipls.filter_and_smooth(meas, prior_mean, prior_cov, cost_fn=cost_fn_ipls)
    results.append((lm_ipls_ms, lm_ipls_Ps, "LM-IPLS"))
    plot_results(states, meas, results)


def plot_results(traj, meas, means_and_covs):
    K = meas.shape[0]
    means_and_covs = [(means, covs.reshape((K, 1)), label) for means, covs, label in means_and_covs]
    _, ax = plt.subplots()
    vis.plot_1d_est(traj, meas, means_and_covs, ax=ax)
    plt.show()


class MeasType(Enum):
    Quadratic = "quadratic"
    Cubic = "cubic"


def parse_args():
    parser = argparse.ArgumentParser(description="LM-IEKS paper experiment.")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--meas_type", type=MeasType, required=True)
    parser.add_argument("--num_iter", type=int, default=3)

    return parser.parse_args()


if __name__ == "__main__":
    main()
