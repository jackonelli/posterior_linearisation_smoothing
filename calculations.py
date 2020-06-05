import numpy as np


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
