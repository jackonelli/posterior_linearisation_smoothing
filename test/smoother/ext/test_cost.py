"""Test (GN-)IEKS
Check that the (GN-)IEKS implementation matches the one in the paper:

"Levenberg-marquardt and line-search extended kalman smoother"

Runs EKS and compares with stored matlab output.
"""
import unittest
from pathlib import Path
import numpy as np
from src.cost_fn.ext import analytical_smoothing_cost, analytical_smoothing_cost_time_dep
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import CoordTurn
from data.lm_ieks_paper.coord_turn_example import get_specific_states_from_file, simulate_data, Type


class TestCost(unittest.TestCase):
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
        meas_model = MultiSensorRange(sensors, R)

        prior_mean = np.array([0, 0, 1, 0, 0])
        prior_cov = np.diag([0.1, 0.1, 1, 1, 1])

        _, measurements, _, ss_ms = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.GN, 10)
        measurements = measurements[:, :2]
        self.assertAlmostEqual(
            analytical_smoothing_cost(ss_ms, measurements, prior_mean, prior_cov, motion_model, meas_model),
            1.039569495177240e03,
        )

    def test_cmp_time_indep_and_time_dep(self):
        dt = 0.01
        time_steps = 2
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

        states, measurements = simulate_data(motion_model, meas_model, prior_mean[:-1], time_steps)
        a = 1 + dt * 10 * np.cumsum(np.random.randn(1, time_steps))
        states = np.column_stack((states, a))
        measurements = measurements[:, :2]
        self.assertAlmostEqual(
            analytical_smoothing_cost(states, measurements, prior_mean, prior_cov, motion_model, meas_model),
            analytical_smoothing_cost_time_dep(states, measurements, prior_mean, prior_cov, motion_model, meas_model),
        )
