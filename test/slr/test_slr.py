import unittest
from functools import partial
import numpy as np
from src.sigma_points import UnscentedTransform
from src.models.nonstationary_growth import NonStationaryGrowth
from src.slr.sigma_points import SigmaPointSlr


class TestSlr(unittest.TestCase):
    def test_ipls_paper_ex(self):
        ungm = NonStationaryGrowth(alpha=0.9, beta=10, gamma=8, delta=1.2, proc_noise=1)
        motion_fn = partial(ungm.map_set, time_step=1)
        ut = UnscentedTransform(1, 0, 1 / 2)
        mean = np.atleast_1d(6.27)
        cov = np.atleast_2d(2.0198)
        sp, weights = ut.sigma_points(mean, cov)
        self.assertTrue(np.allclose(sp, np.array([6.27, 8.01060, 4.52939]).reshape((3, 1))))

        slr_ = SigmaPointSlr(ut)
        z, psi, phi = slr_.slr(motion_fn, mean, cov)
        A, b, O = slr_.linear_params(motion_fn, mean, cov)
        self.assertAlmostEqual(A.item(), 0.648364625788135)
        self.assertAlmostEqual(b.item(), 6.106518639999370)
        self.assertAlmostEqual(O.item(), 0.002780297964241)
