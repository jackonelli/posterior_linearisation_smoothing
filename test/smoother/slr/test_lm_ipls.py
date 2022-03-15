"""Test (GN-)IPLS
Check that the (GN-)IPLS implementation matches the one in the paper:

"Iterated posterior linearisation smoother"

Runs (GN-)IPLS and compares with stored matlab output.
"""
import unittest
from functools import partial
import numpy as np
from src.utils import setup_logger
from src.smoother.slr.ipls import SigmaPointIpls
from src.smoother.slr.lm_ipls import SigmaPointLmIpls
from src.sigma_points import SphericalCubature
from data.ipls_paper.data import get_specific_states_from_file, gen_measurements
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import CoordTurn
from data.lm_ieks_paper.coord_turn_example import simulate_data, get_specific_states_from_file, Type
from src.cost_fn.slr import slr_smoothing_cost_pre_comp


class TestLmIpls(unittest.TestCase):
    def test_lambda_zero_results_in_plain_ipls(self):
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
        meas_model = MultiSensorRange(sensors, R)

        prior_mean = np.array([0, 0, 1, 0, 0])
        prior_cov = np.diag([0.1, 0.1, 1, 1, 1])

        num_iter = 3
        states, measurements = simulate_data(motion_model, meas_model, prior_mean[:-1], 20)

        sigma_point_method = SphericalCubature()
        cost_fn = partial(
            slr_smoothing_cost_pre_comp,
            measurements=measurements,
            m_1_0=prior_mean,
            P_1_0_inv=np.linalg.inv(prior_cov),
        )

        ipls = SigmaPointIpls(motion_model, meas_model, sigma_point_method, num_iter)
        mf, Pf, ms, Ps, _ = ipls.filter_and_smooth(measurements, prior_mean, prior_cov, cost_fn)
        lm_ipls = SigmaPointLmIpls(motion_model, meas_model, sigma_point_method, num_iter, 10, 0.0, 10)
        lm_mf, lm_Pf, lm_ms, lm_Ps, _ = lm_ipls.filter_and_smooth(measurements, prior_mean, prior_cov, cost_fn)
        # lm_ieks = LmIeks(motion_model, meas_model, num_iter, cost_improv_iter_lim, lambda_, nu)
        # lm_mf, lm_Pf, lm_ms, lm_Ps, _iter_cost = lm_ieks.filter_and_smooth_with_init_traj(
        #     measurements, prior_mean, prior_cov, init_traj, 1, cost_fn
        # )
        self.assertTrue(np.allclose(mf, lm_mf))
        self.assertTrue(np.allclose(ms, lm_ms))
