import unittest
import numpy as np
from src.sigma_points import SphericalCubature


class TestSphericalCubature(unittest.TestCase):
    def test_dim_single_state(self):
        mean = np.array([0, 1])
        cov = 2 * np.eye(2)
        sp, weights = SphericalCubature().sigma_points(mean, cov)
        should_be = np.array([2, 1])
        D_x = mean.shape[0]
        self.assertEqual(weights.shape, (2 * D_x,))
        self.assertEqual(sp.shape, (2 * D_x, D_x))
        self.assertTrue(np.allclose(sp[0, :], should_be))
