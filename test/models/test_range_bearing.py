import unittest
from functools import partial
import numpy as np
from src.models.range_bearing import RangeBearing


# TODO: Add numerial test
class TestRangeBearing(unittest.TestCase):
    def test_dim_single_state(self):
        pos = np.array([0, 0])
        R = np.eye(2)
        meas_model = RangeBearing(pos, R)
        state = np.ones((5,))
        meas = meas_model.mapping(state)
        self.assertEqual(meas.shape, (2,))
        # Most important to check that there are no sub-arrays
        self.assertAlmostEqual(meas[0], np.sqrt(2))

    def test_dim_multiple_states(self):
        pos = np.array([0, 0])
        R = np.eye(2)
        meas_model = RangeBearing(pos, R)
        state = np.ones((10, 5))
        meas = meas_model.map_set(state)
        self.assertEqual(meas.shape, (10, 2))
        self.assertAlmostEqual(meas[0, 0], np.sqrt(2))
