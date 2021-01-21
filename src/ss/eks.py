"""Python port of Simo Särkkäs matlab impl

Used for debugging
"""
import numpy as np


def basic_eks(Z, m1, P1, Q, R, f_fun, df_fun, h_fun, dh_fun):
    m = m1
    P = P1
    D_x = m1.shape[0]
    ts_fin = Z.shape[0]
    MM = np.zeros((ts_fin, D_x))
    PP = np.zeros((ts_fin, D_x, D_x))

    for k in range(ts_fin):
        if k > 0:
            f = f_fun(m)
            F = df_fun(m)
            m = f
            P = F @ P @ F.T + Q

        h = h_fun(m)
        H = dh_fun(m)

        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        m = m + K @ (Z[k, :] - h)
        P = P - K @ S @ K.T
        MM[k, :] = m
        PP[k, :, :] = P

    ms = MM[-1, :]
    Ps = PP[-1, :, :]
    MMS = np.empty(MM.shape)
    PPS = np.empty(PP.shape)
    MMS[-1, :] = ms
    PPS[-1, :] = Ps
    for k in np.flip(np.arange(0, ts_fin - 1)):
        m = MM[k, :]
        P = PP[k, :, :]

        f = f_fun(m)
        F = df_fun(m)
        mp = f
        Pp = F @ P @ F.T + Q
        Ck = P @ F.T @ np.linalg.inv(Pp)
        ms = m + Ck @ (ms - mp)
        Ps = P + Ck @ (Ps - Pp) @ Ck.T
        MMS[k, :] = ms
        PPS[k, :, :] = Ps

    print("last k: ", k)

    return MMS, PPS, MM, PP
