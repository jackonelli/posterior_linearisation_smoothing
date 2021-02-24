"""Test (GN-)IEKS
Check that the (GN-)IEKS implementation matches the one in the paper:

"Levenberg-marquardt and line-search extended kalman smoother"

Runs EKS and compares with stored matlab output.
"""
import unittest
from pathlib import Path
from functools import partial
import numpy as np
from src.cost import slr_smoothing_cost, slr_smoothing_cost_pre_comp
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import LmCoordTurn
from src.slr.base import SlrCache
from src.slr.sigma_points import SigmaPointSlr
from src.sigma_points import SphericalCubature
from data.lm_ieks_paper.coord_turn_example import get_specific_states_from_file, Type


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
        motion_model = LmCoordTurn(dt, Q)

        sens_pos_1 = np.array([-1.5, 0.5])
        sens_pos_2 = np.array([1, 1])
        sensors = np.row_stack((sens_pos_1, sens_pos_2))
        std = 0.5
        R = std ** 2 * np.eye(2)
        meas_model = MultiSensorRange(sensors, R)

        prior_mean = np.array([0, 0, 1, 0, 0])
        prior_cov = np.diag([0.1, 0.1, 1, 1, 1])

        _, measurements, _, ss_ms = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.LM, 10)
        measurements = measurements[:, :2]
        K = measurements.shape[0]
        np.random.seed(0)
        covs = np.array([prior_cov] * K) * (0.90 + np.random.rand() / 5)
        slr_cache = SlrCache(motion_model.map_set, meas_model.map_set, SigmaPointSlr(SphericalCubature()))
        new_proto = partial(
            slr_smoothing_cost_pre_comp,
            measurements=measurements,
            m_1_0=prior_mean,
            P_1_0=prior_cov,
        )

        on_the_fly = partial(
            slr_smoothing_cost,
            covs=covs,
            measurements=measurements,
            m_1_0=prior_mean,
            P_1_0=prior_cov,
            motion_model=motion_model,
            meas_model=meas_model,
            slr=SigmaPointSlr(SphericalCubature()),
        )

        slr_cache.update(ss_ms, covs)
        pre_comp = partial(
            new_proto,
            proc_bar=slr_cache.proc_bar,
            meas_bar=slr_cache.meas_bar,
            proc_cov=np.array(
                [err_cov_k + motion_model.proc_noise(k) for k, (_, _, err_cov_k) in enumerate(slr_cache.proc_lin)]
            ),
            meas_cov=np.array(
                [err_cov_k + meas_model.meas_noise(k) for k, (_, _, err_cov_k) in enumerate(slr_cache.meas_lin)]
            ),
        )
        self.assertAlmostEqual(pre_comp(ss_ms), on_the_fly(ss_ms))
        _, measurements, _, ss_ms = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.GN, 1)
        slr_cache.update(ss_ms, covs)
        pre_comp = partial(
            new_proto,
            proc_bar=slr_cache.proc_bar,
            meas_bar=slr_cache.meas_bar,
            proc_cov=np.array(
                [err_cov_k + motion_model.proc_noise(k) for k, (_, _, err_cov_k) in enumerate(slr_cache.proc_lin)]
            ),
            meas_cov=np.array(
                [err_cov_k + meas_model.meas_noise(k) for k, (_, _, err_cov_k) in enumerate(slr_cache.meas_lin)]
            ),
        )
        self.assertAlmostEqual(pre_comp(ss_ms), on_the_fly(ss_ms))
