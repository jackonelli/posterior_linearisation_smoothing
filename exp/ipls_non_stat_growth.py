"""Example: Non-stationary growth model

Reproducing the experiment in the paper:

    "Iteraterd posterior linearization smoother"
"""

import argparse
from enum import Enum
import logging
from functools import partial
from pathlib import Path
import numpy as np
from src import visualization as vis
from src.utils import setup_logger
from src.models.nonstationary_growth import NonStationaryGrowth
from src.models.cubic import Cubic
from src.models.quadratic import Quadratic
from src.sigma_points import UnscentedTransform
from src.smoother.ext.ieks import Ieks
from src.smoother.ext.lm_ieks import LmIeks
from src.filter.iplf import SigmaPointIplf
from src.filter.prlf import SigmaPointPrLf
from src.smoother.slr.ipls import SigmaPointIpls
from src.smoother.slr.lm_ipls import SigmaPointLmIpls
from data.ipls_paper.data import get_specific_states_from_file, gen_measurements, simulate_data
from src.cost import slr_smoothing_cost_pre_comp, analytical_smoothing_cost_time_dep
from src.analytics import rmse
import matplotlib.pyplot as plt
from src.utils import tikz_1d_tab_format, tikz_err_bar_tab_format


def main():
    np.random.seed(1)
    args = parse_args()
    log = logging.getLogger(__name__)
    experiment_name = "ipls"
    setup_logger(f"logs/{experiment_name}.log", logging.DEBUG)
    log.info(f"Running experiment: {experiment_name}")
    K = 50
    D_x = 1
    motion_model = NonStationaryGrowth(alpha=0.9, beta=10, gamma=8, delta=1.2, proc_noise=1)

    meas_model = (
        Cubic(coeff=1 / 20, meas_noise=1) if args.meas_type == MeasType.Cubic else Quadratic(coeff=1 / 20, meas_noise=1)
    )

    prior_mean = np.atleast_1d(5)
    prior_cov = np.atleast_2d([4])

    # MC DATA
    # num_mc_runs = 1000  # 1000 in original exp
    # num_mc_per_traj = 50
    # num_trajs = num_mc_runs // num_mc_per_traj
    # trajs, noise, _, _ = get_specific_states_from_file(Path.cwd() / "data/ipls_paper")
    # assert trajs.shape == (K, num_trajs)
    # assert noise.shape == (K, num_mc_runs)
    # meas = gen_measurements(traj, noise[:, 0], meas_model)

    states, meas = simulate_data(K, prior_mean, prior_cov, motion_model, meas_model)

    # The paper simply states that "[t]he SLRs have been implemented using the unscented transform
    # with N = 2 D_x + 1 sigma-points and the weight of the sigma-point located on the mean is 1/3."

    # The following settings ensures that:
    # a) w_0 (the weight of the mean sigma point) is 1/3
    # b) there is no difference for the weights for the mean and cov estimation
    sigma_point_method = UnscentedTransform(1, 0, 1 / 2)
    assert sigma_point_method.weights(D_x)[0][0] == 1 / 3
    assert np.allclose(sigma_point_method.weights(D_x)[0], sigma_point_method.weights(D_x)[1])

    # traj_idx = _mc_iter_to_traj_idx(0, num_mc_per_traj)
    # traj = trajs[:, traj_idx].reshape((K, 1))
    # traj = traj[:min_K, :]

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
    ms_gn_ieks, Ps_gn_ieks, cost_gn_ieks, rmses_gn_ieks, neeses_gn_ieks = ieks.filter_and_smooth(
        meas,
        prior_mean,
        prior_cov,
        cost_fn_eks,
    )
    results.append(
        (ms_gn_ieks, Ps_gn_ieks, "GN-IEKS"),
    )
    lm_ieks = LmIeks(motion_model, meas_model, args.num_iter, 10, 1e-2, 10)
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
        P_1_0=prior_cov,
    )

    gn_ipls = SigmaPointIpls(motion_model, meas_model, sigma_point_method, args.num_iter)
    _, _, gn_ipls_ms, gn_ipls_Ps, _ = gn_ipls.filter_and_smooth(meas, prior_mean, prior_cov, cost_fn=cost_fn_ipls)

    results.append((gn_ipls_ms, gn_ipls_Ps, "GN-IPLS"))
    lm_ipls = SigmaPointLmIpls(
        motion_model, meas_model, sigma_point_method, args.num_iter, cost_improv_iter_lim=10, lambda_=1e-4, nu=10
    )

    _, _, lm_ipls_ms, lm_ipls_Ps, _ = lm_ipls.filter_and_smooth(meas, prior_mean, prior_cov, cost_fn=cost_fn_ipls)
    results.append((lm_ipls_ms, lm_ipls_Ps, "LM-IPLS"))
    # tikz_results(states, meas, results)
    plot_results(states, meas, results)
    # filter_squared_errs = np.zeros((K, 1))
    # smooth_squared_errs = np.zeros((K, 1))
    # for mc_iter in range(num_mc_runs):
    #     # initialise storage
    #     if mc_iter % 100 == 0:
    #         log.info(f"MC iter: {mc_iter}")
    #     traj_idx = _mc_iter_to_traj_idx(mc_iter, num_mc_per_traj)
    #     traj = trajs[:, traj_idx].reshape((K, 1))
    #     meas = gen_measurements(traj, noise[:, mc_iter])
    #     mf, Pf, ms, Ps, _ = ipls.filter_and_smooth(meas, prior_mean, prior_cov, cost_fn=None)
    #     filter_squared_errs += (mf - traj) ** 2
    #     smooth_squared_errs += (ms - traj) ** 2
    # smooth_rmses = np.sqrt(smooth_squared_errs / num_mc_runs)
    # plt.plot(smooth_rmses)
    # plt.show()

    # mf, Pf = mf.reshape((K,)), Pf.reshape((K,))
    # ms, Pf = mf.reshape((K,)), Pf.reshape((K,))


def tikz_results(traj, meas, means_and_covs):
    dir_ = Path.cwd().parent / "paper/fig/non_stat_growth"
    dir_ = Path.cwd() / "tmp_tikz"
    for data, label in [(traj, "states"), (meas, "meas")]:
        vis.write_to_tikz_file(tikz_1d_tab_format(data.squeeze()), dir_, f"{label.lower()}.data")
    for mean, var, label in means_and_covs:
        vis.write_to_tikz_file(
            tikz_err_bar_tab_format(np.arange(1, len(mean) + 1), mean.squeeze(), np.sqrt(var.squeeze())),
            dir_,
            f"{label.lower()}.data",
        )


def plot_results(traj, meas, means_and_covs):
    K = meas.shape[0]
    means_and_covs = [(means, covs.reshape((K,)), label) for means, covs, label in means_and_covs]
    fig, ax = plt.subplots()
    vis.plot_1d_est(traj, meas, means_and_covs, ax=ax)
    # vis.write_to_tikz_file(vis.to_tikz(fig), Path.cwd(), "nonstat")
    plt.show()


def _mc_iter_to_traj_idx(mc_iter: int, num_mc_per_traj) -> int:
    return int(mc_iter / num_mc_per_traj)


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
