# Demo of the iterated posterior linearisation smoother with the same scenario with
# a univariate non-stationary growth model as in

# �. F. Garc�a-Fern�ndez, L. Svensson and S. S�rkk�, "Iterated Posterior Linearization Smoother,"
# in IEEE Transactions on Automatic Control, vol. 62, no. 4, pp. 2056-2063, April 2017

# First, statistical linear regression (SLR) using IPLF, then smoother and then iterated
# SLRs, with filtering and smoothing
from pathlib import Path
from functools import partial
import math
import numpy as np
from exp.ipls_non_stat_growth import _mc_iter_to_traj_idx
from data.ipls_paper.data import get_specific_states_from_file, gen_measurements
from src.models.nonstationary_growth import NonStationaryGrowth
from src.models.cubic import Cubic
from src.sigma_points import UnscentedTransform
from src.slr.sigma_points import SigmaPointSlr
from src.filter.prlf import SigmaPointPrLf
from src.smoother.slr.prls import SigmaPointPrLs


# randn("seed", 9)
# rand("seed", 9)

# Scenario_ungm_trajectories

# Parametros sigma points
Nx = 1  # Dimension of the state
W0 = 1 / 3  # Weight of sigma-point at the mean
Nz = 1  # Dimension of the measurement
Nsteps = 50
Nmc = 1000
num_mc_per_traj = 50

motion_model = NonStationaryGrowth(alpha=0.9, beta=10, gamma=8, delta=1.2, proc_noise=1)
meas_model = Cubic(coeff=1 / 20, proc_noise=1)
sigma_point_method = UnscentedTransform(1, 0, 1 / 2)
slr = SigmaPointSlr(sigma_point_method)

prlf = SigmaPointPrLf(motion_model, meas_model, sigma_point_method)
prls = SigmaPointPrLs(motion_model, meas_model, sigma_point_method)

X_multi_series, noise_z = get_specific_states_from_file(Path.cwd() / "data/ipls_paper")
X_multi_series = X_multi_series.T
noise_z.T

Wn = (1 - W0) / (2 * Nx)
square_error_t_tot = np.zeros((1, Nsteps))

ruido_mean_ini = np.random.randn(1, Nmc)

nees_t_tot = np.zeros((1, Nsteps))
rms_t_series = np.zeros((1, Nmc))
square_error_t_series = np.zeros((Nmc, Nsteps))

square_error_t_tot_smoothing = np.zeros((1, Nsteps))


# Nit_s-> Number of iterations in smoothing
Nit_s = 10
Nit_iplf = 1  # Number of iterations in filtering (Nit_iplf=1 is similar to the UKF)

square_error_t_tot_smoothing_series = np.zeros((Nmc, Nsteps, Nit_s + 1))  # For filtering and the N_it_s iterations
NLL_smoothing_series = np.zeros((Nmc, Nsteps, Nit_s + 1))  # Negative log-likelihood (See Deisenroth_thesis)
Nees_smoothing_series = np.zeros((Nmc, Nsteps, Nit_s + 1))  # NEES
cte_NLL = Nx / 2 * np.log(2 * math.pi)  # Constant for NLL

x0 = np.atleast_1d(5)
P_ini = np.atleast_2d(4)
mean_ini = x0
Q = motion_model.proc_noise(0)
R = meas_model.meas_noise(0)
chol_R = np.sqrt(R)
a = 1 / 20


# figure(1)
# clf
# plot(X_multi_series.T)

# randn("seed", 9)
# rand("seed", 9)

# Number of Monte Carlo runs
t = 0
for i in range(1, Nmc):

    n_trajectory = _mc_iter_to_traj_idx(i, Nmc // num_mc_per_traj)
    X_multi = X_multi_series[n_trajectory, :]

    meank = mean_ini
    Pk = P_ini

    square_error_t = np.zeros((Nsteps,))
    square_error_t_smoothing = np.zeros((1, Nsteps))

    nees_t = np.zeros((Nsteps,))

    meank_t = np.zeros((Nsteps, Nx))
    Pk_t = np.zeros((Nsteps, Nx, Nx))

    # SLR parameters for dynamics
    A_dyn = np.zeros((Nx, Nx, Nsteps))
    b_dyn = np.zeros((Nx, Nsteps))
    Omega_dyn = np.zeros((Nx, Nx, Nsteps))

    # SLR parameters for measurements
    A_m = np.zeros((Nz, Nx, Nsteps))
    b_m = np.zeros((Nz, Nsteps))
    Omega_m = np.zeros((Nz, Nz, Nsteps))

    # Generation of measurements
    z_real_t = np.zeros((Nz, Nsteps))
    for k in range(Nsteps):
        state_k = X_multi[k]
        # noise=chol_R*noise_z(:,k+Nsteps*(i-1))
        noise = chol_R * noise_z[k, i]
        z_real = a * state_k ** 3 + noise
        z_real_t[:, k] = z_real
    # UKF

    for k in range(Nsteps):
        pos_x = X_multi[k]

        z_real = z_real_t[:, k]

        # Calculate iterated SLR

        meank_j = meank
        Pk_j = Pk

        for p in range(Nit_iplf):

            # SLR of function a*x**3
            # [A_l, b_l, Omega_l] = SLR_measurement_ax3(meank_j, Pk_j, a, weights, W0, Nx, Nz)
            (A_l, b_l, Omega_l) = slr.linear_params(meas_model.map_set, meank_j, Pk_j)

            # KF update
            # NOTE: `k` unused here (I expect it to be at least.)
            (mean_ukf_act, var_ukf_act) = prlf._update(z_real, meank, Pk, R, (A_l, b_l, Omega_l), k)

            meank_j = mean_ukf_act
            Pk_j = var_ukf_act

        A_m[:, :, k] = A_l
        b_m[:, k] = b_l
        Omega_m[:, :, k] = Omega_l

        meank_t[k, :] = mean_ukf_act
        Pk_t[k, :, :] = var_ukf_act

        square_error_t[k] = (mean_ukf_act - pos_x) ** 2
        pos_error = mean_ukf_act - pos_x
        ukf_const_idx = 0  # NOTE: used to be 1
        var_pos_act = var_ukf_act[ukf_const_idx]  #
        # nees_t[k]=state_error.T*inv(var_mp_tukf_act)*state_error
        nees_t[k] = pos_error.T / var_pos_act * pos_error

        # NOTE: ones here should be zero
        square_error_t_tot_smoothing_series[i, k, 1] = square_error_t[k]
        NLL_smoothing_series[i, k, 1] = (
            1 / 2 * np.log(var_ukf_act[ukf_const_idx])
            + 1 / 2 * square_error_t[k] / var_ukf_act[ukf_const_idx]
            + cte_NLL
        )
        Nees_smoothing_series[i, k, 1] = square_error_t[k] / var_ukf_act[ukf_const_idx]

        # Prediction
        # [A_dyn_k, b_dyn_k, Omega_dyn_k] = SLR_ungm_dynamic(
        #     mean_ukf_act, var_ukf_act, alfa_mod, beta_mod, gamma_mod, weights, W0, Nx, k
        # )
        # NOTE: might be shifted with one here
        (A_dyn_k, b_dyn_k, Omega_dyn_k) = slr.linear_params(
            partial(motion_model.map_set, time_step=k),
            mean_ukf_act,
            var_ukf_act,
        )

        meank = A_dyn_k * mean_ukf_act + b_dyn_k
        Pk = A_dyn_k * var_ukf_act * A_dyn_k.T + Omega_dyn_k + Q
        # Pk = (Pk + Pk.T) / 2

        A_dyn[:, :, k] = A_dyn_k
        b_dyn[:, k] = b_dyn_k
        Omega_dyn[:, :, k] = Omega_dyn_k

    # Smoother
    # [meank_smoothed_t, Pk_smoothed_t] = linear_rts_smoother(meank_t, Pk_t, A_dyn, b_dyn, Omega_dyn, Q)
    [meank_smoothed_t, Pk_smoothed_t] = prls.smooth_seq_pre_comp_filter(meank_t, Pk_t, A_dyn, b_dyn, Omega_dyn, Q)

    # for k=1:Nsteps
    #     pos_x=X_multi[1,k)
    #     square_error_t_tot_smoothing_series[i,k,2)=(meank_smoothed_t[1,k)-pos_x)**2
    #     NLL_smoothing_series[i,k,2)=1/2*log(Pk_smoothed_t[1,1,k))+1/2*square_error_t_tot_smoothing_series(i,k,2)/Pk_smoothed_t(1,1,k)+cte_NLL
    #     Nees_smoothing_series(i,k,2)=square_error_t_tot_smoothing_series(i,k,2)/Pk_smoothed_t(1,1,k)

    for p in range(1, Nit_s - 1):

        # Iterated SLR using the current posterior

        for k in range(1, Nsteps):
            # Generation of sigma points
            meank = meank_smoothed_t[:, k]
            Pk = Pk_smoothed_t[:, :, k]

            # SLR for measurement
            # [A_l, b_l, Omega_l] = SLR_measurement_ax3(meank, Pk, a, weights, W0, Nx, Nz)
            (A_l, b_l, Omega_l) = slr.linear_params(meas_model.map_set, meank_j, Pk_j)

            A_m[:, :, k] = A_l
            b_m[:, k] = b_l
            Omega_m[:, :, k] = Omega_l

            # SLR for dynamics
            # [A_dyn_k, b_dyn_k, Omega_dyn_k] = SLR_ungm_dynamic(
            #     meank, Pk, alfa_mod, beta_mod, gamma_mod, weights, W0, Nx, k
            # )
            # NOTE: might be shifted with one here
            (A_dyn_k, b_dyn_k, Omega_dyn_k) = slr.linear_params(
                partial(motion_model.map_set, time_step=k),
                mean_ukf_act,
                var_ukf_act,
            )

            A_dyn[:, :, k] = A_dyn_k
            b_dyn[:, k] = b_dyn_k
            Omega_dyn[:, :, k] = Omega_dyn_k

        # Filter with the linearised model

        [meank_t, Pk_t] = linear_kf_full(mean_ini, P_ini, A_m, b_m, Omega_m, A_dyn, b_dyn, Omega_dyn, R, Q, z_real_t)

        # Smoother

        [meank_smoothed_t, Pk_smoothed_t] = linear_rts_smoother(meank_t, Pk_t, A_dyn, b_dyn, Omega_dyn, Q)

        # for k in range(1,Nsteps):
        #     pos_x=X_multi[1,k)
        #     square_error_t_tot_smoothing_series[i,k,p+2)=(meank_smoothed_t[1,k)-pos_x)**2
        #     NLL_smoothing_series[i,k,p+2)=1/2*log(Pk_smoothed_t[1,1,k))+1/2*square_error_t_tot_smoothing_series[i,k,p+2)/Pk_smoothed_t(1,1,k)+cte_NLL
        #     Nees_smoothing_series(i,k,p+2)=square_error_t_tot_smoothing_series(i,k,p+2)/Pk_smoothed_t(1,1,k)
    # Square error calculation

    # for k=1:Nsteps
    #     pos_x=X_multi(1,k)
    #     square_error_t_smoothing(k)=(meank_smoothed_t(1,k)-pos_x)**2
    square_error_t_tot = square_error_t_tot + square_error_t
    square_error_t_series[i, :] = square_error_t
    square_error_t_tot_smoothing = square_error_t_tot_smoothing + square_error_t_smoothing

    nees_t_tot = nees_t_tot + nees_t
    rms_t_series[i] = np.sqrt(sum(square_error_t) / (Nsteps))

    # display(["Completed iteration no ", num2str(i)," time ", num2str(t), " seconds"])

square_error_t_tot = square_error_t_tot / Nmc
rmse_filtering_tot = np.sqrt(sum(square_error_t_tot) / (Nsteps))

square_error_t_tot_smoothing = square_error_t_tot_smoothing / Nmc

rmse_tot_smoothing = np.sqrt(sum(square_error_t_tot_smoothing) / (Nsteps))

rmse_t_tot_smoothing_series = np.sqrt(sum(np.squeeze(sum(square_error_t_tot_smoothing_series, 1)), 1) / (Nmc * Nsteps))


nees_t_tot = nees_t_tot / Nmc


# Smoothing error for different J
rmse_smoothing_1 = np.sqrt(sum(square_error_t_tot_smoothing_series[:, :, 2], 1) / Nmc)
rmse_smoothing_5 = np.sqrt(sum(square_error_t_tot_smoothing_series[:, :, 6], 1) / Nmc)
rmse_smoothing_10 = np.sqrt(sum(square_error_t_tot_smoothing_series[:, :, 11], 1) / Nmc)


# Output figure IPLS(i)-J denotes a IPLS with i SLR iterations for the IPLF and
# J SLR smoothing iterations)


NLL_average_list = np.zeros((1, Nit_s))

for i in range(1, Nit_s + 1):
    NLL = NLL_smoothing_series[:, :, i]

    NLL_average = sum(sum(NLL)) / (Nsteps * Nmc)
    NLL_average_list[i] = NLL_average
