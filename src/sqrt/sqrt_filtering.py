"""Square root implementation of the Kalman filter (KF)"""
import logging
import numpy as np

LOGGER = logging.getLogger(__name__)


def _predict(x_kminus1_kminus1, P_sqrt_kminus1_kminus1, linearization):
    """Square root KF prediction step

    Args:
        x_kminus1_kminus1: x_{k-1 | k-1}
        P_kminus1_kminus1: P_{k-1 | k-1}
        linearization (tuple): (A, b, Q_sqrt) param's for linear (affine) approx

    Returns:
        x_k_kminus1: x_{k | k-1}
        P_sqrt_k_kminus1: P^{1/2}_{k | k-1}
        aux ((G, Z)): Auxiliary variables for later use during smoothing.
    """
    (D_x,) = x_kminus1_kminus1.shape

    A, b, Q_sqrt = linearization

    x_k_kminus1 = A @ x_kminus1_kminus1 + b

    instr_mat = np.block([[Q_sqrt, A @ P_sqrt_kminus1_kminus1], [np.zeros((D_x, D_x)), P_sqrt_kminus1_kminus1]])
    _, R_l_transp = np.linalg.qr(instr_mat.T)

    P_sqrt_k_kminus1, Y_p, Z_p = R_l_transp.T[:D_x, :D_x], R_l_transp.T[D_x:, :D_x], R_l_transp.T[D_x:, D_x : 2 * D_x]
    G = Y_p @ np.linalg.inv(P_sqrt_k_kminus1)
    return x_k_kminus1, P_sqrt_k_kminus1, (G, Z_p)


def _update(y_k, x_k_kminus1, P_sqrt_k_kminus1, linearization):
    """Square root KF update step
    Args:
        y_k
        x_k_kminus1: x_{k | k-1}
        P_k_kminus1: P_{k | k-1}
        linearization (tuple): (H, c, R_sqrt) param's for linear (affine) approx

    Returns:
        x_k_k: x_{k | k}
        P_sqrt_k_k: P^{1/2}_{k | k}
    """
    H, c, R_sqrt = linearization
    (D_y, D_x) = H.shape
    instr_mat = np.block([[R_sqrt, H @ P_sqrt_k_kminus1], [np.zeros((D_x, D_y)), P_sqrt_k_kminus1]])
    _, R_l_transp = np.linalg.qr(instr_mat.T)
    # Extract blocks from R matrix
    X_u, Y_u, P_sqrt_k_k = R_l_transp.T[:D_y, :D_y], R_l_transp.T[D_y:, :D_y], R_l_transp.T[D_y:, D_y : (D_y + D_x)]
    K = Y_u @ np.linalg.inv(X_u)

    y_mean = H @ x_k_kminus1 + c
    x_k_k = x_k_kminus1 + (K @ (y_k - y_mean)).reshape(x_k_kminus1.shape)
    return x_k_k, P_sqrt_k_k
