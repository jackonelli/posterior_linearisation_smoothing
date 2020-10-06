import unittest
import numpy as np
from src.filtering import _predict as ref_predict
from src.sqrt_filtering import _predict as sqrt_predict
from src.analytics import is_pos_def
from hypothesis.extra.numpy import arrays
from hypothesis import given
from hypothesis.strategies import composite, floats, integers


@composite
def square_matrix(draw, elements=floats(-10, 10)):
    dim = draw(integers(2, 10))
    return draw(arrays(np.float, (dim, dim), elements=floats(-10, 10)))


class SqrtPredict(unittest.TestCase):
    @given(P_sqrt=square_matrix())
    def test_compare_ref(self, P_sqrt):
        D_x = P_sqrt.shape[0]
        x = np.random.rand(D_x)
        # P_sqrt = np.random.rand(D_x, D_x)
        P = P_sqrt @ P_sqrt.T

        A = np.random.rand(D_x, D_x)
        b = np.random.rand(D_x)
        Q_sqrt = np.random.rand(D_x, D_x)
        Q = Q_sqrt @ Q_sqrt.T

        ref_lin = (A, b, Q)
        sqrt_lin = (A, b, Q_sqrt)

        x_ref, P_ref = ref_predict(x, P, ref_lin)
        x_sqrt, P_sqrt_new = sqrt_predict(x, P_sqrt, sqrt_lin)

        self.assertTrue(np.allclose(x_ref, x_sqrt))

        self.assertTrue(np.allclose(P_ref, P_sqrt_new @ P_sqrt_new.T))


if __name__ == "__main__":
    unittest.main()
