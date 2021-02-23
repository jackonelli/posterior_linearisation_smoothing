"""Test LM-IEKS
Check that the LM-IEKS implementation matches the one in the paper:

"Levenberg-marquardt and line-search extended kalman smoother"

Runs LM-IEKS and compares with stored matlab output.
"""
import unittest
from pathlib import Path
from functools import partial
import numpy as np
from src.smoother.ext.lm_ieks import LmIeks
from src.cost import analytical_smoothing_cost
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import LmCoordTurn
from src.analytics import nees
from data.lm_ieks_paper.coord_turn_example import get_specific_states_from_file, Type


class TestLmIeks(unittest.TestCase):
    def test_cmp_with_ss_impl(self):
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

        num_iter = 1
        states, measurements, ss_mf, ss_ms = get_specific_states_from_file(
            Path.cwd() / "data/lm_ieks_paper", Type.LM, num_iter
        )
        measurements = measurements[:, :2]
        lambda_ = 1e-2
        nu = 10
        cost_improv_iter_lim = 10

        cost_fn = partial(
            analytical_smoothing_cost,
            meas=measurements,
            m_1_0=prior_mean,
            P_1_0=prior_cov,
            motion_model=motion_model,
            meas_model=meas_model,
        )

        K = measurements.shape[0]
        init_traj = (np.zeros((K, prior_mean.shape[0])), None)

        ieks = LmIeks(motion_model, meas_model, num_iter, cost_improv_iter_lim, lambda_, nu)
        mf, Pf, ms, Ps, _iter_cost = ieks.filter_and_smooth_with_init_traj(
            measurements, prior_mean, prior_cov, init_traj, 1, cost_fn
        )
        self.assertTrue(np.allclose(mf, ss_mf))
        self.assertTrue(np.allclose(ms, ss_ms))

        num_iter = 10
        _, measurements, ss_mf, ss_ms = get_specific_states_from_file(
            Path.cwd() / "data/lm_ieks_paper", Type.LM, num_iter
        )
        measurements = measurements[:, :2]

        ieks = LmIeks(motion_model, meas_model, num_iter, cost_improv_iter_lim, lambda_, nu)
        mf, Pf, ms, Ps, _iter_cost = ieks.filter_and_smooth_with_init_traj(
            measurements, prior_mean, prior_cov, init_traj, 1, cost_fn
        )
        self.assertTrue(np.allclose(mf, ss_mf))
        self.assertTrue(np.allclose(ms, ss_ms))
        # Summation over the time steps and columns of the cov seq.
        matlab_covs_sum = np.array([1.8843, 0.3417, 20.4884, 2.2148, 498.6999])
        self.assertTrue(np.allclose(Ps.sum(0).sum(1), matlab_covs_sum, rtol=1e-4, atol=1e-4))
        calc_nees = np.mean(nees(states, ms[:, :-1], Ps[:, :-1, :-1]))
        # self.assertAlmostEqual(calc_nees, 2.2094012168057)
