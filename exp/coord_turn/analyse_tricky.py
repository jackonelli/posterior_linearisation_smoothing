"""Analysing a specific case where the LM-IPLS fails"""

import logging
import argparse
from pathlib import Path
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from src import visualization as vis
from src.filter.ekf import Ekf
from src.smoother.ext.eks import Eks
from src.smoother.ext.ieks import Ieks
from src.smoother.ext.lm_ieks import LmIeks
from src.smoother.ext.ls_ieks import LsIeks
from src.smoother.slr.ipls import SigmaPointIpls
from src.smoother.slr.ls_ipls import SigmaPointLsIpls
from src.smoother.slr.lm_ipls import SigmaPointLmIpls
from src.line_search import GridSearch
from src.slr.sigma_points import SigmaPointSlr
from src.sigma_points import SphericalCubature
from src.cost import analytical_smoothing_cost, slr_smoothing_cost_pre_comp, slr_noop_cost, slr_smoothing_cost_means
from src.utils import setup_logger
from src.analytics import rmse
from src.visualization import to_tikz, write_to_tikz_file
from src.models.range_bearing import MultiSensorRange, MultiSensorBearings
from src.models.coord_turn import CoordTurn
from data.lm_ieks_paper.coord_turn_example import Type, get_specific_states_from_file, simulate_data
from exp.coord_turn.common import (
    MeasType,
    run_smoothing,
    calc_iter_metrics,
    mc_stats,
    plot_results,
    plot_cost,
    plot_neeses,
    plot_rmses,
)


def main():
    args = parse_args()
    log = logging.getLogger(__name__)
    experiment_name = "analyse_tricky_ct"
    setup_logger(f"logs/{experiment_name}.log", logging.DEBUG)
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

    lambda_ = 1e-2

    num_iter = args.num_iter
    meas_model = MultiSensorBearings(sensors, R)
    data_dir = Path.cwd() / "data/lm_ieks_paper/tricky"
    states = np.genfromtxt(data_dir / "lm_div_states.csv", dtype=float)
    measurements = np.genfromtxt(data_dir / "lm_div_meas.csv", dtype=float)

    estimates = []
    costs = []
    neeses = []
    rmses = []
    cost_fn_eks = partial(
        analytical_smoothing_cost,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
        motion_model=motion_model,
        meas_model=meas_model,
    )

    ms_ieks, Ps_ieks, cost_ieks, rmses_ieks, neeses_ieks = run_smoothing(
        Ieks(motion_model, meas_model, num_iter),
        states,
        measurements,
        prior_mean,
        prior_cov,
        cost_fn_eks,
    )
    estimates.append(
        (ms_ieks, Ps_ieks, cost_ieks[1:], "IEKS"),
    )
    costs.append(
        (cost_ieks, "IEKS"),
    )
    rmses.append(
        (rmses_ieks, "IEKS"),
    )
    neeses.append(
        (neeses_ieks, "IEKS"),
    )

    ms_lm_ieks, Ps_lm_ieks, cost_lm_ieks, rmses_lm_ieks, neeses_lm_ieks = run_smoothing(
        LmIeks(motion_model, meas_model, num_iter, 10, lambda_, 10),
        states,
        measurements,
        prior_mean,
        prior_cov,
        cost_fn_eks,
    )
    estimates.append(
        (ms_lm_ieks, Ps_lm_ieks, cost_lm_ieks[1:], "LM-IEKS"),
    )
    costs.append(
        (cost_lm_ieks, "LM-IEKS"),
    )
    rmses.append(
        (rmses_lm_ieks, "LM-IEKS"),
    )
    neeses.append(
        (neeses_lm_ieks, "LM-IEKS"),
    )

    ms_ls_ieks, Ps_ls_ieks, cost_ls_ieks, rmses_ls_ieks, neeses_ls_ieks = run_smoothing(
        LsIeks(motion_model, meas_model, num_iter, GridSearch(cost_fn_eks, 20)),
        states,
        measurements,
        prior_mean,
        prior_cov,
        cost_fn_eks,
    )
    estimates.append(
        (ms_ls_ieks, Ps_ls_ieks, cost_ls_ieks[1:], "LS-IEKS"),
    )
    costs.append(
        (cost_ls_ieks, "LS-IEKS"),
    )
    rmses.append(
        (rmses_ls_ieks, "LS-IEKS"),
    )
    neeses.append(
        (neeses_ls_ieks, "LS-IEKS"),
    )

    sigma_point_method = SphericalCubature()
    ls_cost_fn = partial(
        slr_smoothing_cost_means,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
        motion_fn=motion_model.map_set,
        meas_fn=meas_model.map_set,
        slr_method=SigmaPointSlr(sigma_point_method),
    )

    pre_comp_cost = partial(
        slr_smoothing_cost_pre_comp,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
    )
    ms_ipls, Ps_ipls, cost_ipls, rmses_ipls, neeses_ipls = run_smoothing(
        SigmaPointIpls(motion_model, meas_model, sigma_point_method, num_iter),
        states,
        measurements,
        prior_mean,
        prior_cov,
        pre_comp_cost,
    )
    estimates.append(
        (ms_ipls, Ps_ipls, cost_ipls, "IPLS"),
    )
    costs.append(
        (cost_ipls, "IPLS"),
    )
    rmses.append(
        (rmses_ipls, "IPLS"),
    )
    neeses.append(
        (neeses_ipls, "IPLS"),
    )

    ms_lm_ipls, Ps_lm_ipls, cost_lm_ipls, rmses_lm_ipls, neeses_lm_ipls = run_smoothing(
        SigmaPointLmIpls(
            motion_model, meas_model, sigma_point_method, num_iter, cost_improv_iter_lim=10, lambda_=lambda_, nu=10
        ),
        states,
        measurements,
        prior_mean,
        prior_cov,
        pre_comp_cost,
    )
    estimates.append(
        (ms_lm_ipls, Ps_lm_ipls, cost_lm_ipls, "LM-IPLS"),
    )
    costs.append(
        (cost_lm_ipls, "LM-IPLS"),
    )
    rmses.append(
        (rmses_lm_ipls, "LM-IPLS"),
    )
    neeses.append(
        (neeses_lm_ipls, "LM-IPLS"),
    )

    ms_ls_ipls, Ps_ls_ipls, cost_ls_ipls, rmses_ls_ipls, neeses_ls_ipls = run_smoothing(
        SigmaPointLsIpls(motion_model, meas_model, sigma_point_method, num_iter, GridSearch, 10),
        states,
        measurements,
        prior_mean,
        prior_cov,
        ls_cost_fn,
    )
    estimates.append(
        (ms_ls_ipls, Ps_ls_ipls, cost_ls_ipls, "LS-IPLS"),
    )
    costs.append(
        (cost_ls_ipls, "LS-IPLS"),
    )
    rmses.append(
        (rmses_ls_ipls, "LS-IPLS"),
    )
    neeses.append(
        (neeses_ls_ipls, "LS-IPLS"),
    )

    plot_cost(costs)
    plot_neeses(neeses)
    plot_rmses(rmses)
    plot_results(
        states,
        estimates,
        None,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="LM-IEKS paper experiment.")
    parser.add_argument("--num_iter", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    main()
