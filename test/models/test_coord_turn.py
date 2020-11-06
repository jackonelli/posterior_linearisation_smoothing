import unittest
from functools import partial
import numpy as np
from src.models.coord_turn import CoordTurn

# TODO: Add numerial test
class TestCoordTurn(unittest.TestCase):
    def test_dim_single_state(self):
        sampling_period = 0.1
        motion_model = CoordTurn(sampling_period, None)
        state = np.zeros((5,))
        pred_state = motion_model.mapping(state)
        self.assertEqual(pred_state.shape, state.shape)
        self.assertAlmostEqual(pred_state[0], 0)

    def test_dim_multiple_states(self):
        sampling_period = 0.1
        motion_model = CoordTurn(sampling_period, None)
        state = np.zeros((10, 5))
        pred_state = motion_model.map_set(state)
        self.assertEqual(pred_state.shape, state.shape)
        self.assertAlmostEqual(pred_state[0, 0], 0)
