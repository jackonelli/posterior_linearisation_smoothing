"""Test (GN-)IEKS
Check that the (GN-)IEKS implementation matches the one in the paper:

"Levenberg-marquardt and line-search extended kalman smoother"

Runs EKS and compares with stored matlab output.
"""
import unittest
from pathlib import Path
from functools import partial
import numpy as np
from src.cost_fn.slr import slr_smoothing_cost, slr_smoothing_cost_pre_comp, slr_smoothing_cost_means
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import CoordTurn
from src.slr.base import SlrCache
from src.slr.sigma_points import SigmaPointSlr
from src.sigma_points import SphericalCubature
from data.lm_ieks_paper.coord_turn_example import get_specific_states_from_file, Type


class TestCost(unittest.TestCase):
    def test_cmp_slr_costs(self):
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

        _, measurements, _, ss_ms = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.LM, 10)
        measurements = measurements[:, :2]
        K = measurements.shape[0]
        np.random.seed(0)
        covs = np.array([prior_cov] * K) * (0.90 + np.random.rand() / 5)
        slr_cache = SlrCache(motion_model, meas_model, SigmaPointSlr(SphericalCubature()))
        pre_comp_proto = partial(
            slr_smoothing_cost_pre_comp,
            measurements=measurements,
            m_1_0=prior_mean,
            P_1_0_inv=np.linalg.inv(prior_cov),
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

        varying_means_proto = partial(
            slr_smoothing_cost_means,
            measurements=measurements,
            m_1_0=prior_mean,
            P_1_0_inv=np.linalg.inv(prior_cov),
            estimated_covs=covs,
            motion_fn=motion_model.map_set,
            meas_fn=meas_model.map_set,
            slr_method=SigmaPointSlr(SphericalCubature()),
        )
        varying_means = partial(
            varying_means_proto,
            motion_cov_inv=slr_cache.proc_cov_inv,
            meas_cov_inv=slr_cache.meas_cov_inv,
        )

        proc_cov_inv, meas_cov_inv = slr_cache.inv_cov()
        pre_comp = partial(
            pre_comp_proto,
            motion_bar=slr_cache.proc_bar,
            meas_bar=slr_cache.meas_bar,
            motion_cov_inv=proc_cov_inv,
            meas_cov_inv=meas_cov_inv,
        )

        self.assertAlmostEqual(pre_comp(ss_ms), on_the_fly(ss_ms))
        self.assertAlmostEqual(pre_comp(ss_ms), varying_means(ss_ms))

        _, measurements, _, ss_ms = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.GN, 1)
        slr_cache.update(ss_ms, covs)
        pre_comp = partial(
            pre_comp_proto,
            motion_bar=slr_cache.proc_bar,
            meas_bar=slr_cache.meas_bar,
            motion_cov_inv=slr_cache.proc_cov_inv,
            meas_cov_inv=slr_cache.meas_cov_inv,
        )
        varying_means = partial(
            varying_means_proto,
            motion_cov_inv=slr_cache.proc_cov_inv,
            meas_cov_inv=slr_cache.meas_cov_inv,
        )

        self.assertAlmostEqual(pre_comp(ss_ms), on_the_fly(ss_ms))
        self.assertAlmostEqual(pre_comp(ss_ms), varying_means(ss_ms))
