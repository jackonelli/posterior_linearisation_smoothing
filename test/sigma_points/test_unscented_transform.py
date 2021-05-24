import unittest
import numpy as np
from src.sigma_points import UnscentedTransform


class TestUnscentedTransform(unittest.TestCase):
    def test_dim_single_state(self):
        mean = np.array([0, 1])
        cov = 2 * np.eye(2)
        sp, weights = UnscentedTransform(1, 2, 3).gen_sigma_points(mean, cov)
        D_x = mean.shape[0]
        self.assertEqual(weights.shape, (2 * D_x + 1,))
        self.assertEqual(sp.shape, (2 * D_x + 1, D_x))
        self.assertTrue(np.allclose(sp[0, :], mean))
