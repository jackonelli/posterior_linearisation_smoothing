"""Test inexact line search LS-IEKS
Check that the LS-IEKS implementation matches the one in the paper:

"Levenberg-marquardt and line-search extended kalman smoother"

Runs LS-IEKS and compares with stored matlab output.
"""
import unittest
from pathlib import Path
from functools import partial
import numpy as np
from src.smoother.ext.ieks import Ieks
from src.cost import analytical_smoothing_cost, grad_analytical_smoothing_cost, _ss_dir_der_analytical_smoothing_cost
from src.models.range_bearing import MultiSensorBearings
from src.models.coord_turn import CoordTurn
from src.line_search import GridSearch
from data.lm_ieks_paper.coord_turn_example import get_specific_states_from_file, Type

import matplotlib.pyplot as plt


class TestInexactLsIeks(unittest.TestCase):
    def test_cmp_dir_der_with_ss_impl(self):
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

        num_iter = 1
        _, measurements, ss_mf, ss_ms = get_specific_states_from_file(
            Path.cwd() / "data/lm_ieks_paper", Type.GN, num_iter
        )
        measurements = measurements[:, 2:]

        K = measurements.shape[0]
        D_x = prior_mean.shape[0]
        init_traj = (np.zeros((K, D_x)), None)
        search_dir = np.ones((K, D_x))

        # From SS matlab impl
        # He does not use scale the cost with 1/2, making the ref value twice bigger than our target.
        ref_val = 4.423579239781962e02
        comp_dir_dev = _ss_dir_der_analytical_smoothing_cost(
            init_traj[0], search_dir, measurements, prior_mean, prior_cov, motion_model, meas_model
        )
        self.assertAlmostEqual(ref_val, 2 * comp_dir_dev)

        comp_grad = grad_analytical_smoothing_cost(
            init_traj[0], measurements, prior_mean, prior_cov, motion_model, meas_model
        )

        search_dir = np.ones((K * D_x,))
        self.assertAlmostEqual(ref_val, comp_grad.T @ search_dir * 2)
