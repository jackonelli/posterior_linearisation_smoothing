"""Example: Iterative post. lin. smoothing with affine models"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from src.scipy.stats import multivariate_normal as mvn
from src.iterative import iterative_post_lin_smooth
from src.smoothing import rts_smoothing
from src.slr.distributions import Gaussian
from src.slr.slr import Slr
from src.linearizer import Identity
from src.models.affine import Affine
from src.scipy.linalg import sqrtm, qr
from src.analytics import pos_def_check


def main():
    prior_mean = np.array([1, 1, 3, 2])
    prior_cov = 1 * np.eye(4)
    T = 1
    A = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]])
    b = 0 * np.ones((4,))
    Q = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1.5, 0],
            [0, 0, 0, 1.5],
        ]
    )
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    c = np.zeros((H @ prior_mean).shape)
    R = 2 * np.eye(2)
    P_sqrt = sqrtm(prior_cov)
    D_x = 4
    instr_mat = np.block([[sqrtm(Q), A @ P_sqrt], [np.zeros((D_x, D_x)), P_sqrt]])
    print(instr_mat.shape)
    _, R_l = qr(instr_mat)
    R_l_transp = R_l.T
    np.set_printoptions(precision=2)

    # (xs_slr, Ps_slr, xf_slr, Pf_slr, linearizations) = iterative_post_lin_smooth(
    #    y, prior_mean, prior_cov, motion_lin, meas_lin, num_iterations
    # )


if __name__ == "__main__":
    main()
