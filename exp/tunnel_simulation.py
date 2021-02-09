"""Experiment: Tunnel simulation

Testing robustness of smoothers in a simulated scenario
where a car passes through a tunnel, thereby going through the stages of

- Starting with low noise measurements (before the tunnel)
- Increased uncertainty while in the tunnel
(- Ending past the tunnel, again with certain measurements)
"""
import logging
from pathlib import Path
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from src import visualization as vis
from src.filter.ekf import Ekf
from src.smoother.ext.eks import Eks
from src.smoother.ext.ieks import Ieks
from src.smoother.ext.lm_ieks import LmIeks
from src.smoother.slr.ipls import SigmaPointIpls
from src.utils import setup_logger
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import LmCoordTurn
from data.lm_ieks_paper.coord_turn_example import simulate_data
from src.smoother.slr.reg_ipls import SigmaPointRegIpls
from src.slr.sigma_points import SigmaPointSlr
from src.sigma_points import SphericalCubature
from src.cost import analytical_smoothing_cost, slr_smoothing_cost
from exp.lm_ieks_coord_turn import plot_results, plot_cost, gn_ipls, gn_ieks, lm_ipls, lm_ieks


def main():
    log = logging.getLogger(__name__)
    experiment_name = "tunnel_simulation"
    setup_logger(f"logs/{experiment_name}.log", logging.INFO)
    log.info(f"Running experiment: {experiment_name}")
    # seed = 0
    # np.random.seed(seed)
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

    states, measurements = simulate_data(sens_pos_1, sens_pos_2, std, dt, prior_mean[:-1], time_steps=500)

    num_iter = 10

    results = []
    cost_fn_eks = partial(
        analytical_smoothing_cost,
        meas=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
        motion_model=motion_model,
        meas_model=meas_model,
    )
    ms_gn_ieks, Ps_gn_ieks, cost_gn_ieks = gn_ieks(
        motion_model, meas_model, num_iter, measurements, prior_mean, prior_cov, cost_fn_eks
    )
    results.append((ms_gn_ieks, Ps_gn_ieks, cost_gn_ieks[1:], "GN-IEKS"))
    ms_lm_ieks, Ps_lm_ieks, cost_lm_ieks = lm_ieks(
        motion_model, meas_model, num_iter, measurements, prior_mean, prior_cov, cost_fn_eks
    )
    results.append((ms_lm_ieks, Ps_lm_ieks, cost_lm_ieks[1:], "LM-IEKS"))

    sigma_point_method = SphericalCubature()
    cost_fn_ipls = partial(
        slr_smoothing_cost,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
        motion_model=motion_model,
        meas_model=meas_model,
        slr=SigmaPointSlr(sigma_point_method),
    )
    ms_gn_ipls, Ps_gn_ipls, cost_gn_ipls = gn_ipls(
        motion_model,
        meas_model,
        sigma_point_method,
        num_iter,
        measurements,
        prior_mean,
        prior_cov,
        cost_fn_ipls,
    )
    results.append((ms_gn_ipls, Ps_gn_ipls, cost_gn_ipls[1:], "GN-IPLS"))
    ms_lm_ipls, Ps_lm_ipls, cost_lm_ipls = lm_ipls(
        motion_model, meas_model, sigma_point_method, num_iter, measurements, prior_mean, prior_cov, cost_fn_ipls
    )
    results.append((ms_lm_ipls, Ps_lm_ipls, cost_lm_ipls[1:], "LM-IPLS"))
    plot_results(
        states,
        results,
    )


if __name__ == "__main__":
    main()
