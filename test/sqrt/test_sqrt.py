import unittest
import numpy as np
from src import filtering as filter_ref
from src import sqrt_filtering as filter_sqrt
from hypothesis.extra.numpy import arrays
from hypothesis import given
from hypothesis.strategies import composite, floats, integers


def square_mat_strat(dim, elements=floats(-10, 10)):
    return arrays(np.float, (dim, dim), elements=elements)


def vec_strat(dim, elements=floats(-10, 10)):
    return arrays(np.float, (dim,), elements=elements)


@composite
def predict_data(draw, elements=floats(-10, 10)):
    """Strategy for test of sqrt predict step"""
    dim = draw(integers(1, 10))
    P_sqrt, Q_sqrt, A = draw(square_mat_strat(dim)), draw(square_mat_strat(dim)), draw(square_mat_strat(dim))
    P_sqrt += 1.5 * np.eye(dim)
    A += 1 * np.eye(dim)
    x, b = draw(vec_strat(dim)), draw(vec_strat(dim))
    return x, A, b, Q_sqrt, P_sqrt


@composite
def update_data(draw, elements=floats(0.1, 10)):
    """Strategy for test of sqrt predict step"""
    dim_x = draw(integers(1, 10))
    dim_y = draw(integers(1, 10))
    P_sqrt, R_sqrt = draw(square_mat_strat(dim_x, elements)), draw(square_mat_strat(dim_y, elements))
    # Decrease risk of sing matrix
    R_sqrt += 1.5 * np.eye(dim_y)
    P_sqrt += 1.5 * np.eye(dim_x)
    H = draw(arrays(np.float, (dim_y, dim_x), elements=elements))
    x, y, c = draw(vec_strat(dim_x)), draw(vec_strat(dim_y)), draw(vec_strat(dim_y))
    return x, y, H, c, R_sqrt, P_sqrt


def is_not_singular(sample):
    _, _, H, _, R_sqrt, P_sqrt = sample
    P = P_sqrt @ P_sqrt.T
    R = R_sqrt @ R_sqrt.T
    S = H @ P @ H.T + R
    return np.linalg.matrix_rank(S) == S.shape[0]


class SqrtImpl(unittest.TestCase):
    @given(data=predict_data())
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

    @given(data=update_data().filter(is_not_singular))
    def test_update_compare_ref(self, data):
        x, y, H, c, R_sqrt, P_sqrt = data
        P = P_sqrt @ P_sqrt.T
        R = R_sqrt @ R_sqrt.T

        ref_lin = (H, c, R)
        sqrt_lin = (H, c, R_sqrt)

        x_ref, P_ref = filter_ref._update(y, x, P, ref_lin)
        x_sqrt, P_sqrt_new = filter_sqrt._update(y, x, P_sqrt, sqrt_lin)

        self.assertTrue(np.allclose(x_ref, x_sqrt))
        self.assertTrue(np.allclose(P_ref, P_sqrt_new @ P_sqrt_new.T, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
