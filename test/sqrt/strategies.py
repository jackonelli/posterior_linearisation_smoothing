"""Common strategies for hypothesis"""
import numpy as np
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats


def square_mat_strat(dim, elements=floats(-10, 10)):
    return arrays(np.float, (dim, dim), elements=elements)


def vec_strat(dim, elements=floats(-10, 10)):
    return arrays(np.float, (dim,), elements=elements)
