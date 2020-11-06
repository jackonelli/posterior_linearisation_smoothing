import unittest
from functools import partial
import numpy as np
from src.sigma_points import SphericalCubature


class TestCoordTurn(unittest.TestCase):
    def test_dim_single_state(self):
        mean = np.array([0, 1])
        cov = 2 * np.eye(2)
        sp, _ = SphericalCubature().sigma_points(mean, cov)
        should_be = np.array([2, 1])
        self.assertTrue(np.allclose(sp[0, :], should_be))
