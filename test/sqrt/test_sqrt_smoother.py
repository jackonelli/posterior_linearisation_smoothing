import unittest
import numpy as np
from hypothesis.extra.numpy import arrays
from hypothesis import given
from hypothesis.strategies import composite, floats, integers
from src import filtering as filter_ref
from src import sqrt_filtering as filter_sqrt
from test.sqrt.strategies import square_mat_strat, vec_strat


@composite
def smooth_data(draw, elements=floats(-10, 10)):
    """Strategy for test of sqrt predict step"""
    dim_x, dim_y = draw(integers(1, 10)), draw(integers(1, 10))
    x_base = draw(vec_strat(dim_x))
    x_noise = vec_strat(dim_x, elements=floats(-0.5, 0.5))
    x_k_K = x_k_K, x_k_kminus1
    dim = draw(integers(1, 10))
    P_sqrt, Q_sqrt, A = draw(square_mat_strat(dim)), draw(square_mat_strat(dim)), draw(square_mat_strat(dim))
    P_sqrt += 1.5 * np.eye(dim)
    A += 1 * np.eye(dim)
    x, b = draw(vec_strat(dim)), draw(vec_strat(dim))
    return x, A, b, Q_sqrt, P_sqrt


def is_not_singular(sample):
    _, _, H, _, R_sqrt, P_sqrt = sample
    P = P_sqrt @ P_sqrt.T
    R = R_sqrt @ R_sqrt.T
    S = H @ P @ H.T + R
    return np.linalg.matrix_rank(S) == S.shape[0]


class SqrtSmoother(unittest.TestCase):
    @given(data=smooth_data())
    def test_predict_compare_ref(self, data):
        x, A, b, Q_sqrt, P_sqrt = data
        P = P_sqrt @ P_sqrt.T

        Q = Q_sqrt @ Q_sqrt.T

        ref_lin = (A, b, Q)
        sqrt_lin = (A, b, Q_sqrt)

        x_ref, P_ref = filter_ref._predict(x, P, ref_lin)
        x_sqrt, P_sqrt_new, _ = filter_sqrt._predict(x, P_sqrt, sqrt_lin)

        self.assertTrue(np.allclose(x_ref, x_sqrt))

        self.assertTrue(np.allclose(P_ref, P_sqrt_new @ P_sqrt_new.T))


if __name__ == "__main__":
    unittest.main()
