from timeit import timeit
from functools import partial
from pathlib import Path
import numpy as np
from src.cost import slr_smoothing_cost, slr_smoothing_cost_pre_comp
from src.sigma_points import SphericalCubature
from src.slr.sigma_points import SigmaPointSlr
from src.slr.base import SlrCache
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import CoordTurn
from data.lm_ieks_paper.coord_turn_example import get_specific_states_from_file, Type


def main():
    dt = 0.01
    qc = 0.01
    qw = 10
    Q = np.array(
        [
            [qc * dt ** 3 / 3, 0, qc * dt ** 2 / 2, 0, 0],
            [0, qc * dt ** 3 / 3, 0, qc * dt ** 2 / 2, 0],
            [qc * dt ** 2 / 2, 0, qc * dt, 0, 0],
            [0, qc * dt ** 2 / 2, 0, qc * dt, 0],
            [0, 0, 0, 0, dt * qw],
        ]
    )
    motion_model = CoordTurn(dt, Q)

    sens_pos_1 = np.array([-1.5, 0.5])
    sens_pos_2 = np.array([1, 1])
    sensors = np.row_stack((sens_pos_1, sens_pos_2))
    std = 0.5
    R = std ** 2 * np.eye(2)
    meas_model = MultiSensorRange(sensors, R)

    prior_mean = np.array([0, 0, 1, 0, 0])
    prior_cov = np.diag([0.1, 0.1, 1, 1, 1])

    _, measurements, _, ss_ms = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.GN, 10)
    measurements = measurements[:, :2]
    K = measurements.shape[0]
    covs = np.array([prior_cov] * K)
    slr_cache = SlrCache(motion_model.map_set, meas_model.map_set, SigmaPointSlr(SphericalCubature()))
    slr_cache.update(ss_ms, covs)
    new_proto = partial(
        slr_smoothing_cost_pre_comp,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
    )
    new = partial(
        new_proto,
        traj=ss_ms,
        proc_bar=slr_cache.proc_bar,
        meas_bar=slr_cache.meas_bar,
        proc_cov=np.array(
            [err_cov_k + motion_model.proc_noise(k) for k, (_, _, err_cov_k) in enumerate(slr_cache.proc_lin)]
        ),
        meas_cov=np.array(
            [err_cov_k + meas_model.meas_noise(k) for k, (_, _, err_cov_k) in enumerate(slr_cache.meas_lin)]
        ),
    )
    old = partial(
        slr_smoothing_cost,
        ss_ms,
        covs,
        measurements,
        prior_mean,
        prior_cov,
        motion_model,
        meas_model,
        SigmaPointSlr(SphericalCubature()),
    )
    num_samples = 10
    old_time = timeit(old, number=num_samples) / num_samples
    print(old_time)
    new_time = timeit(new, number=num_samples) / num_samples
    print(new_time)
    assert np.allclose(new(), old())


if __name__ == "__main__":
    main()
