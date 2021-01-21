"""Python port of Simo Särkkäs matlab impl

Used for debugging
"""
import numpy as np


def gn_ieks(measurements, prior_mean, prior_cov, Q, R, f_fun, df_fun, h_fun, dh_fun, niter, MN0):
    MN = MN0
    # JJ = np.zeros((niter, 1))
    # J = ss_cost(MN,Z,m1,P1,Q,R,f_fun,h_fun)
    D_x = prior_mean.shape[0]
    ts_fin = measurements.shape[0]
    for iter_ in range(1, niter + 1):
        print("Matlab iter: ", iter_)
        m = prior_mean
        P = prior_cov
        xf = np.zeros((ts_fin, D_x))
        Pf = np.zeros((ts_fin, D_x, D_x))
        for k in range(ts_fin):
            if k > 0:
                # Fix the nominal distribution for prediction
                mn = MN[k - 1, :]

                F = df_fun(mn)
                f = f_fun(mn) + F @ (m - mn)

                m = f
                P = F @ P @ F.T + Q
            mn = MN[k, :]

            H = dh_fun(mn)
            h = h_fun(mn) + H @ (m - mn)

            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            m = m + K @ (measurements[k, :] - h)
            P = P - K @ S @ K.T

            xf[k, :] = m
            Pf[k, :, :] = P
        ms_K = xf[-1, :]
        Ps_K = Pf[-1, :, :]

        xs = np.empty(xf.shape)
        Ps = np.empty(Pf.shape)
        xs[-1, :] = ms_K
        Ps[-1, :, :] = Ps_K

        for k in np.flip(np.arange(0, ts_fin - 1)):
            m = xf[k, :]
            P = Pf[k, :, :]

            mn = MN[k, :]

            F = df_fun(mn)
            f = f_fun(mn) + F @ (m - mn)

            mp = f
            Pp = F @ P @ F.T + Q
            Ck = P @ F.T @ np.linalg.inv(Pp)

            ms_K = m + Ck @ (ms_K - mp)
            Ps_K = P + Ck @ (Ps_K - Pp) @ Ck.T
            xs[k, :] = ms_K
            Ps[k, :, :] = Ps_K

        # % No line search
        MN = xs.copy()
    return xf, Pf, xs, Ps
