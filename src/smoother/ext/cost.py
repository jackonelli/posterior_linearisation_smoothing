import numpy as np


def cost(means, measurements, m_1_0, P_1_0, motion_model, meas_model):
    diff = means[0, :] - m_1_0
    _cost = diff.T @ np.linalg.inv(P_1_0) @ diff
    for k in range(0, means.shape[0] - 1):
        diff = means[k + 1, :] - motion_model.mapping(means[k, :])
        _cost += diff.T @ np.linalg.inv(motion_model.proc_noise(k)) @ diff
        # measurements are zero indexed, i.e. k-1 --> y_k
        diff = measurements[k, :] - meas_model.mapping(means[k, :])
        _cost += diff.T @ np.linalg.inv(meas_model.meas_noise(k)) @ diff
    diff = measurements[-1, :] - meas_model.mapping(means[-1, :])
    _cost += diff.T @ np.linalg.inv(meas_model.meas_noise(k)) @ diff

    return _cost


def ml_cost(means, measurements, m_1_0, P_1_0, motion_model, meas_model, Q, R):
    diff = means[0, :] - m_1_0
    _cost = diff.T @ np.linalg.inv(P_1_0) @ diff
    for k in range(0, means.shape[0] - 1):
        diff = means[k + 1, :] - motion_model(means[k, :])
        _cost += diff.T @ np.linalg.inv(Q) @ diff
        # measurements are zero indexed, i.e. k-1 --> y_k
        diff = measurements[k, :] - meas_model(means[k, :])
        _cost += diff.T @ np.linalg.inv(R) @ diff
    diff = measurements[-1, :] - meas_model(means[-1, :])
    _cost += diff.T @ np.linalg.inv(R) @ diff

    return _cost
