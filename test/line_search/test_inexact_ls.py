"""Test inexact line search LS-IEKS
Check that the LS-IEKS implementation matches the one in the paper:

"Levenberg-marquardt and line-search extended kalman smoother"

Runs LS-IEKS and compares with stored matlab output.
"""
import unittest
from functools import partial
from pathlib import Path
import numpy as np
from src.smoother.ext.ieks import Ieks
from src.cost_fn.ext import (
    analytical_smoothing_cost,
    grad_analytical_smoothing_cost,
    dir_der_analytical_smoothing_cost,
)
from src.models.range_bearing import MultiSensorBearings
from src.models.coord_turn import CoordTurn
from src.line_search import GridSearch, ArmijoLineSearch
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

        # Test with simple search dir
        # From SS matlab impl
        # He does not use scale the cost with 1/2, making the ref value twice bigger than our target.
        search_dir = np.ones((K, D_x))
        ref_val = 4.423579239781962e02
        comp_dir_dev = dir_der_analytical_smoothing_cost(
            init_traj[0], search_dir, measurements, prior_mean, prior_cov, motion_model, meas_model
        )
        self.assertAlmostEqual(ref_val, 2 * comp_dir_dev)

        # comp_grad = grad_analytical_smoothing_cost(
        #     init_traj[0], measurements, prior_mean, prior_cov, motion_model, meas_model
        # )

        # search_dir = np.ones((K * D_x,))
        # self.assertAlmostEqual(ref_val, search_dir.T @ comp_grad * 2)

        # Test with actual search dir
        cost_fn = partial(
            analytical_smoothing_cost,
            measurements=measurements,
            m_1_0=prior_mean,
            P_1_0=prior_cov,
            motion_model=motion_model,
            meas_model=meas_model,
        )

        m_0 = init_traj[0]
        ieks = Ieks(motion_model, meas_model, num_iter)
        _, _, m_1, _, _ = ieks.filter_and_smooth_with_init_traj(
            measurements, prior_mean, prior_cov, init_traj, 1, cost_fn
        )
        # Sanity check that we use the correct points to test the line search.
        self.assertAlmostEqual(m_1.sum(), -23.808794681191973)

        search_dir = m_1 - m_0
        ref_val = -6.406867572871646e02
        comp_dir_dev = dir_der_analytical_smoothing_cost(
            m_0, search_dir, measurements, prior_mean, prior_cov, motion_model, meas_model
        )
        self.assertAlmostEqual(ref_val, 2 * comp_dir_dev)

        # comp_grad = grad_analytical_smoothing_cost(m_0, measurements, prior_mean, prior_cov, motion_model, meas_model)
        # self.assertAlmostEqual(ref_val, search_dir.flatten() @ comp_grad * 2)

        _, _, m_2, _, _ = ieks.filter_and_smooth_with_init_traj(
            measurements, prior_mean, prior_cov, init_traj, 1, cost_fn
        )

        # Sanity check that we use the correct points to test the line search.
        self.assertAlmostEqual(m_2.sum(), 4.551162489010954e03)
        search_dir = m_2 - m_1
        ref_val = -42.612770542340790
        comp_dir_dev = dir_der_analytical_smoothing_cost(
            m_1, search_dir, measurements, prior_mean, prior_cov, motion_model, meas_model
        )
        self.assertAlmostEqual(ref_val, 2 * comp_dir_dev)
