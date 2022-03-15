"""Test (GN-)IEKS
Check that the (GN-)IEKS implementation matches the one in the paper:

"Levenberg-marquardt and line-search extended kalman smoother"

Runs (GN-)IEKS and compares with stored matlab output.
"""
import unittest
from pathlib import Path
from functools import partial
import numpy as np
from src.smoother.ext.ls_ieks import LsIeks
from src.line_search import GridSearch, ArmijoLineSearch
from src.cost_fn.ext import analytical_smoothing_cost, dir_der_analytical_smoothing_cost
from src.models.range_bearing import MultiSensorBearings
from src.models.coord_turn import CoordTurn
from data.lm_ieks_paper.coord_turn_example import get_specific_states_from_file, Type


class TestExactLsIeks(unittest.TestCase):
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
        motion_model = CoordTurn(dt, Q)

        sens_pos_1 = np.array([-1.5, 0.5])
        sens_pos_2 = np.array([1, 1])
        sensors = np.row_stack((sens_pos_1, sens_pos_2))
        std = 0.5
        R = std ** 2 * np.eye(2)
        meas_model = MultiSensorBearings(sensors, R)

        prior_mean = np.array([0, 0, 1, 0, 0])
        prior_cov = np.diag([0.1, 0.1, 1, 1, 1])

        num_iter = 10
        _, measurements, _, _ = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.GN, num_iter)
        measurements = measurements[:, 2:]

        K = measurements.shape[0]
        init_traj = (np.zeros((K, prior_mean.shape[0])), None)

        cost_fn = partial(
            analytical_smoothing_cost,
            measurements=measurements,
            m_1_0=prior_mean,
            P_1_0=prior_cov,
            motion_model=motion_model,
            meas_model=meas_model,
        )

        ls_method = GridSearch(cost_fn, 10)
        ls_ieks = LsIeks(motion_model, meas_model, num_iter, ls_method)
        mf, Pf, ms, Ps, _iter_cost = ls_ieks.filter_and_smooth_with_init_traj(
            measurements, prior_mean, prior_cov, init_traj, 1, cost_fn
        )
        self.assertAlmostEqual(ms.sum(), 1223.9428357356326, places=2)


class TestArmijoLsIeks(unittest.TestCase):
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
        motion_model = CoordTurn(dt, Q)

        sens_pos_1 = np.array([-1.5, 0.5])
        sens_pos_2 = np.array([1, 1])
        sensors = np.row_stack((sens_pos_1, sens_pos_2))
        std = 0.5
        R = std ** 2 * np.eye(2)
        meas_model = MultiSensorBearings(sensors, R)

        prior_mean = np.array([0, 0, 1, 0, 0])
        prior_cov = np.diag([0.1, 0.1, 1, 1, 1])

        num_iter = 10
        _, measurements, _, _ = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.GN, num_iter)
        measurements = measurements[:, 2:]

        K = measurements.shape[0]
        init_traj = (np.zeros((K, prior_mean.shape[0])), None)

        cost_fn = partial(
            analytical_smoothing_cost,
            measurements=measurements,
            m_1_0=prior_mean,
            P_1_0=prior_cov,
            motion_model=motion_model,
            meas_model=meas_model,
        )

        dir_der_fn = partial(
            dir_der_analytical_smoothing_cost,
            measurements=measurements,
            m_1_0=prior_mean,
            P_1_0=prior_cov,
            motion_model=motion_model,
            meas_model=meas_model,
        )

        ls_method = ArmijoLineSearch(cost_fn, dir_der_fn, c_1=0.1)
        ls_ieks = LsIeks(motion_model, meas_model, num_iter, ls_method)
        mf, Pf, ms, Ps, _iter_cost = ls_ieks.filter_and_smooth_with_init_traj(
            measurements, prior_mean, prior_cov, init_traj, 1, cost_fn
        )
        self.assertAlmostEqual(ms.sum(), 1.223943641943313e03)
