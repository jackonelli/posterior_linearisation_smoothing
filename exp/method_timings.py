"""Timings of smoother methods"""

from timeit import timeit
from functools import partial
import argparse
import logging
from pathlib import Path
from functools import partial
import numpy as np
from src.smoother.ext.ieks import Ieks
from src.smoother.ext.lm_ieks import LmIeks
from src.smoother.ext.ls_ieks import LsIeks
from src.line_search import ArmijoWolfeLineSearch
from src.smoother.slr.ipls import SigmaPointIpls
from src.smoother.slr.lm_ipls import SigmaPointLmIpls
from src.smoother.slr.ls_ipls import SigmaPointLsIpls
from src.slr.sigma_points import SigmaPointSlr
from src.sigma_points import SphericalCubature
from src.cost_fn.slr import (
    slr_smoothing_cost_pre_comp,
    slr_smoothing_cost_means,
    slr_noop_cost,
)
from src.cost_fn.ext import analytical_smoothing_cost, dir_der_analytical_smoothing_cost, noop_cost
from src.utils import setup_logger
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import CoordTurn
from data.lm_ieks_paper.coord_turn_example import Type, get_specific_states_from_file


def main():
    args = parse_args()
    log = logging.getLogger(__name__)
    experiment_name = "method_timings"
    setup_logger(f"logs/{experiment_name}.log", logging.WARNING)
    log.info(f"Running experiment: {experiment_name}")

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
    motion_model = CoordTurn(dt, Q)

    sens_pos_1 = np.array([-1.5, 0.5])
    sens_pos_2 = np.array([1, 1])
    sensors = np.row_stack((sens_pos_1, sens_pos_2))
    std = 0.5
    R = std ** 2 * np.eye(2)

    prior_mean = np.array([0, 0, 1, 0, 0])
    prior_cov = np.diag([0.1, 0.1, 1, 1, 1])

    num_iter = 1
    np.random.seed(0)
    _, all_meas, _, xs_ss = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.LM, num_iter)
    K = all_meas.shape[0]
    covs = np.array([prior_cov] * K) * (0.90 + np.random.rand() / 5)

    meas_model = MultiSensorRange(sensors, R)
    measurements = all_meas[:, :2]

    cost_fn_eks = partial(
        analytical_smoothing_cost,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
        motion_model=motion_model,
        meas_model=meas_model,
    )

    dir_der_eks = partial(
        dir_der_analytical_smoothing_cost,
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
    time_ieks = partial(
        Ieks(motion_model, meas_model, num_iter).filter_and_smooth_with_init_traj,
        measurements,
        prior_mean,
        prior_cov,
        (xs_ss, covs),
        1,
        noop_cost,
    )

    time_lm_ieks = partial(
        LmIeks(motion_model, meas_model, num_iter, 10, 1e-2, 10).filter_and_smooth_with_init_traj,
        measurements,
        prior_mean,
        prior_cov,
        (xs_ss, covs),
        1,
        cost_fn_eks,
    )

    time_ls_ieks = partial(
        LsIeks(
            motion_model,
            meas_model,
            num_iter,
            ArmijoWolfeLineSearch(cost_fn_eks, dir_der_eks, c_1=0.1, c_2=0.2),
        ).filter_and_smooth_with_init_traj,
        measurements,
        prior_mean,
        prior_cov,
        (xs_ss, covs),
        1,
        cost_fn_eks,
    )
    time_ipls = partial(
        SigmaPointIpls(motion_model, meas_model, sigma_point_method, num_iter).filter_and_smooth_with_init_traj,
        measurements,
        prior_mean,
        prior_cov,
        (xs_ss, covs),
        1,
        slr_noop_cost,
    )
    time_lm_ipls = partial(
        SigmaPointLmIpls(
            motion_model, meas_model, sigma_point_method, num_iter, cost_improv_iter_lim=10, lambda_=1e-2, nu=10
        ).filter_and_smooth_with_init_traj,
        measurements,
        prior_mean,
        prior_cov,
        (xs_ss, covs),
        1,
        cost_fn_ipls,
    )

    cost_fn_ls_ipls = partial(
        slr_smoothing_cost_means,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0_inv=np.linalg.inv(prior_cov),
        motion_fn=motion_model.map_set,
        meas_fn=meas_model.map_set,
        slr_method=SigmaPointSlr(sigma_point_method),
    )
    time_ls_ipls = partial(
        SigmaPointLsIpls(
            motion_model, meas_model, sigma_point_method, num_iter, partial(ArmijoWolfeLineSearch, c_1=0.1, c_2=0.9)
        ).filter_and_smooth_with_init_traj,
        measurements,
        prior_mean,
        prior_cov,
        (xs_ss, covs),
        1,
        cost_fn_ls_ipls,
    )
    num_trials = args.num_trials
    time_ieks = timeit(time_ieks, number=num_trials) / (num_iter * num_trials)
    time_lm_ieks = timeit(time_lm_ieks, number=num_trials) / (num_iter * num_trials)
    time_ls_ieks = timeit(time_ls_ieks, number=num_trials) / (num_iter * num_trials)
    time_ipls = timeit(time_ipls, number=num_trials) / (num_iter * num_trials)
    time_lm_ipls = timeit(time_lm_ipls, number=num_trials) / (num_iter * num_trials)
    time_ls_ipls = timeit(time_ls_ipls, number=num_trials) / (num_iter * num_trials)
    print(f"IEKS: {time_ieks:.2f} s, 100.0%")
    print(f"LM-IEKS: {time_lm_ieks:.2f} s, {time_lm_ieks/time_ieks*100:.2f}%")
    print(f"LS-IEKS: {time_ls_ieks:.2f} s, {time_ls_ieks/time_ieks*100:.2f}%")
    print(f"IPLS: {time_ipls:.2f} s, {time_ipls/time_ieks*100:.2f}%")
    print(f"LM-IPLS: {time_lm_ipls:.2f} s, {time_lm_ipls/time_ieks*100:.2f}%")
    print(f"LS-IPLS: {time_ls_ipls:.2f} s, {time_ls_ipls/time_ieks*100:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="LM-IEKS paper experiment.")
    parser.add_argument("--num_trials", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    main()
