"""Python port of Simo Särkkäs matlab impl

Used for debugging
"""
import numpy as np
from src.smoother.ext.cost import ss_cost, cost
from src.smoother.ext.lm_ieks import _LmIekf, LmIeks


def hybrid_lm_ieks(measurements, prior_mean, prior_cov, motion_model, meas_model, num_iter, init_traj, scaled=False):
    Q = motion_model.proc_noise(None)
    R = meas_model.meas_noise(None)

    f_fun, df_fun = motion_model.mapping, motion_model.jacobian
    h_fun, dh_fun = meas_model.mapping, meas_model.jacobian
    current_ms = init_traj
    # JJ = np.zeros((niter, 1))

    J = cost(current_ms, measurements, prior_mean, prior_cov, motion_model, meas_model)
    print("Init cost:", J)
    D_x = prior_mean.shape[0]
    K = measurements.shape[0]

    lambda_ = 1e-2
    nu = 10

    smoother = LmIeks(motion_model, meas_model, num_iter, lambda_, nu)
    smoother._update_estimates(current_ms)

    for iter_ in range(1, num_iter + 1):
        done = False
        j = 0
        while not done and j < 10:
            smoother._lambda = lambda_
            xf, Pf, xp, Pp = smoother._filter_seq(measurements, prior_mean, prior_cov)
            K = xf.shape[0]
            xs, Ps = smoother._init_smooth_estimates(xf[-1, :], Pf[-1, :, :], K)
            for k in np.flip(np.arange(1, K)):
                #     # RTS update
                #     G_k = P_kminus1_kminus1 @ A.T @ np.linalg.inv(P_k_kminus1)
                #     m_kminus1_K = m_kminus1_kminus1 + G_k @ (m_k_K - m_k_kminus1)
                #     P_kminus1_K = P_kminus1_kminus1 + G_k @ (P_k_K - P_k_kminus1) @ G_k.T
                #     xs[k - 1, :] = m_kminus1_K
                #     Ps[k - 1, :, :] = P_kminus1_K

                # ms_K = xf[-1, :]
                # Ps_K = Pf[-1, :, :]
                # xs = np.empty(xf.shape)
                # Ps = np.empty(Pf.shape)
                # xs[-1, :] = ms_K
                # Ps[-1, :, :] = Ps_K

                # for k in np.flip(np.arange(1, K)):
                m_kminus1_kminus1, P_kminus1_kminus1 = xf[k - 1, :], Pf[k - 1, :, :]
                m_k_kminus1, P_k_kminus1 = xp[k, :], Pp[k, :, :]
                m_k_K, P_k_K = xs[k, :], Ps[k, :, :]
                A, _, Q = smoother._motion_lin(m_kminus1_kminus1, P_kminus1_kminus1, k - 1)
                G_k = P_kminus1_kminus1 @ A.T @ np.linalg.inv(P_k_kminus1)

                x_kminus1_K = m_kminus1_kminus1 + G_k @ (m_k_K - m_k_kminus1)
                P_kminus1_K = P_kminus1_kminus1 + G_k @ (P_k_K - P_k_kminus1) @ G_k.T
                xs[k - 1, :] = x_kminus1_K
                Ps[k - 1, :, :] = P_kminus1_K

            Jp = cost(xs, measurements, prior_mean, prior_cov, motion_model, meas_model)
            print(f"Cost: {Jp}, lambda: {lambda_}, iter: {iter_}, inner: {j}")
            if Jp < J:
                lambda_ /= nu
                done = True
            else:
                lambda_ *= nu
            j += 1
        J = Jp

        # % No line search
        current_ms = xs.copy()
        smoother._update_estimates(current_ms)
    return xf, Pf, xs, Ps
