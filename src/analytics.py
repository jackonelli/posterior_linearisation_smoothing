"""Analytics"""
import numpy as np


def rmse(true, est):
    """Calculate RMSE
    Scalar RMSE value for the whole sequence

    Args:
        true (K, D_x)
        est (K, D_x)

    Returns:
        rmse (float)
    """
    err = true - est
    return np.sqrt((err ** 2).sum(1).mean())


def nees(true, est, cov):
    """Calculate NEES
    Normalized estimation error squared (NEES) for all timesteps

    eps_k = e_k^T P_k^-1 e_k

    Args:
        true (K, D_x)
        est (K, D_x)
        cov (K, D_x, D_x)

    Returns:
        nees (K, 1)
    """
    K, D_x = true.shape
    err = true - est
    nees_ = np.empty((K, 1))
    for k, (err_k, cov_k) in enumerate(zip(err, cov)):
        err_k = err_k.reshape((D_x, 1))
        nees_[k] = _single_nees(err_k, cov_k)
    return nees_


def mc_stats(data):
    num_mc_samples = data.shape[0]
    return np.mean(data, 0), np.std(data, 0) / np.sqrt(num_mc_samples)


def _single_nees(err, cov):
    """Calculate NEES for single entry"""
    return err.T @ np.linalg.inv(cov) @ err


def is_pos_def(x):
    """Is positive definite
    Returns True if all eigen values are positive,
    otherwise False
    """
    return np.all(np.linalg.eigvals(x) > 0)


def calc_subspace_proj_matrix(num_dim: int):
    det_dir = np.ones((num_dim, 1))
    Q, _ = np.linalg.qr(det_dir, mode="complete")
    U = Q[:, 1:]
    return U @ U.T


def make_pos_def(x, eps=0.1):
    u, s, vh = np.linalg.svd(x, hermitian=True)
    neg_sing_vals = s < 0
    s_hat = s * np.logical_not(neg_sing_vals) + eps * neg_sing_vals
    return np.dot(u, np.dot(np.diag(s_hat), vh)), s
