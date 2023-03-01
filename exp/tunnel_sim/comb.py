"""Experiment: Tunnel simulation

Same setup as the realisation script,
but we include to extra trajectories:
    - (IPLS+IEKS): IEKS started at the converged IPLS traj.
    - (IEKS+IPLS): IPLS started at the converged IEKS traj.

Testing robustness of smoothers in a simulated scenario
where a car passes through a tunnel, thereby going through the stages of

- Starting with low noise measurements (before the tunnel)
- Increased uncertainty while in the tunnel
(- Ending past the tunnel, again with certain measurements)
"""
import argparse
import logging
from functools import partial
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from src.smoother.ext.ieks import Ieks
from src.smoother.slr.ipls import SigmaPointIpls
from src.utils import setup_logger
from src.models.range_bearing import RangeBearing
from src.models.coord_turn import CoordTurn
from src.sigma_points import SphericalCubature
from src.cost_fn.ext import analytical_smoothing_cost
from src.cost_fn.slr import slr_smoothing_cost_pre_comp
from exp.coord_turn.common import plot_results, calc_iter_metrics
from src.analytics import rmse, nees
from src.models.range_bearing import to_cartesian_coords
from data.tunnel_traj import get_states_and_meas


def main():
    args = parse_args()
    log = logging.getLogger(__name__)
    experiment_name = "tunnel_simulation"
    setup_logger(f"logs/{experiment_name}.log", logging.DEBUG)
    log.info(f"Running experiment: {experiment_name}")

    if not args.random:
        seed = 2
        log.info(f"Setting seed {seed}")
        np.random.seed(seed)

    # Motion model
    sampling_period = 0.1
    v_scale = 7
    omega_scale = 15
    sigma_v = v_scale * 1
    sigma_omega = omega_scale * np.pi / 180
    eps = 0.1
    Q = np.diag([eps, eps, sampling_period * sigma_v ** 2, eps, sampling_period * sigma_omega ** 2])
    motion_model = CoordTurn(sampling_period, Q)

    # Meas model
    pos = np.array([100, -100])
    # sigma_r = 2
    # sigma_phi = 0.5 * np.pi / 180
    noise_factor = 4
    sigma_r = 2 * noise_factor
    sigma_phi = noise_factor * 0.5 * np.pi / 180

    R = np.diag([sigma_r ** 2, sigma_phi ** 2])
    meas_model = RangeBearing(pos, R)

    # Generate data
    range_ = (0, None)
    tunnel_segment = [140, 175]
    # tunnel_segment = [None, None]
    states, measurements = get_states_and_meas(meas_model, R, range_, tunnel_segment)
    measurements = [meas for meas in measurements]
    cartes_meas = np.apply_along_axis(partial(to_cartesian_coords, pos=pos), 1, measurements)

    prior_mean = np.array([0, 0, 1, 0, 0])
    prior_cov = np.diag([0.1, 0.1, 1, 1, 1])

    results = []
    cost_fn_eks = partial(
        analytical_smoothing_cost,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
        motion_model=motion_model,
        meas_model=meas_model,
    )

    sigma_point_method = SphericalCubature()

    cost_fn_ipls = partial(
        slr_smoothing_cost_pre_comp,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0_inv=np.linalg.inv(prior_cov),
    )
    ms_ieks, Ps_ieks, cost_ieks, rmses_ieks, neeses_ieks = run_smoothing(
        Ieks(motion_model, meas_model, args.num_iter), states, measurements, prior_mean, prior_cov, cost_fn_eks
    )
    results.append((ms_ieks, Ps_ieks, cost_ieks[1:], "IEKS"))

    ms_ipls, Ps_ipls, cost_ipls, rmses_ipls, neeses_ipls = run_smoothing(
        SigmaPointIpls(motion_model, meas_model, sigma_point_method, args.num_iter),
        states,
        measurements,
        prior_mean,
        prior_cov,
        cost_fn_ipls,
    )
    results.append((ms_ipls, Ps_ipls, cost_ipls[1:], "IPLS"))

    ipls_traj = (deepcopy(ms_ipls), deepcopy(Ps_ipls))
    ms_comb, Ps_comb, cost_comb, rmse_comb, nees_comb = run_smoothing(
        Ieks(motion_model, meas_model, args.num_iter),
        states,
        measurements,
        prior_mean,
        prior_cov,
        cost_fn_eks,
        ipls_traj,
    )
    results.append(
        (ms_comb, Ps_comb, cost_comb, "IPLS + IEKS"),
    )

    ieks_traj = (deepcopy(ms_ieks), deepcopy(Ps_ieks))
    ms_comb_2, Ps_comb_2, cost_comb, rmse_comb, nees_comb = run_smoothing(
        SigmaPointIpls(motion_model, meas_model, sigma_point_method, args.num_iter),
        states,
        measurements,
        prior_mean,
        prior_cov,
        cost_fn_ipls,
        ieks_traj,
    )
    results.append(
        (ms_comb_2, Ps_comb_2, cost_comb, "IEKS+IPLS"),
    )

    plot_results(
        states,
        results,
        cartes_meas,
        skip_cov=10,
    )


def run_smoothing(smoother, states, measurements, prior_mean, prior_cov, cost_fn, init_traj=None):
    """Common function that runs a smoother and collects metrics

    Some iterative smoothers may return early if they exceed the limit on the number of loss-improving trials.
    In those cases, the metrics are extended with the last element to a list of length `smoother.num_iter`
    """
    if init_traj is not None:
        _, _, ms, Ps, iter_cost = smoother.filter_and_smooth_with_init_traj(
            measurements, prior_mean, prior_cov, init_traj, 1, cost_fn
        )
        stored_est = smoother.stored_estimates()
        next(stored_est)
        stored_est = list(stored_est)
    else:
        _, _, ms, Ps, iter_cost = smoother.filter_and_smooth(measurements, prior_mean, prior_cov, cost_fn)
        stored_est = list(smoother.stored_estimates())
    rmses = calc_iter_metrics(
        lambda means, covs, states: rmse(means[:, :2], states), stored_est, states, smoother.num_iter
    )
    # assert np.allclose(ms_st, ms)
    neeses = calc_iter_metrics(
        lambda means, covs, states: np.mean(nees(states, means[:, :2], covs[:, :2, :2])),
        stored_est,
        states,
        smoother.num_iter,
    )
    return ms, Ps, iter_cost, rmses, neeses


def parse_args():
    parser = argparse.ArgumentParser(description="Tunnel sim. experiment")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--num_iter", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    main()
