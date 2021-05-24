"""Test (GN-)IPLS
Check that the (GN-)IPLS implementation matches the one in the paper:

"Iterated posterior linearisation smoother"

Runs (GN-)IPLS and compares with stored matlab output.
"""
import unittest
from functools import partial
from pathlib import Path
import numpy as np
from src import visualization as vis
from src.utils import setup_logger
from src.models.nonstationary_growth import NonStationaryGrowth
from src.models.cubic import Cubic
from src.models.quadratic import Quadratic
from src.sigma_points import UnscentedTransform
from src.filter.iplf import SigmaPointIplf
from src.filter.prlf import SigmaPointPrLf
from src.smoother.slr.ipls import SigmaPointIpls
from src.smoother.slr.lm_ipls import SigmaPointLmIpls
from src.smoother.slr.ipls import SigmaPointIpls
from data.ipls_paper.data import get_specific_states_from_file, gen_measurements
from src.cost import slr_smoothing_cost_pre_comp
from src import visualization as vis
from src.analytics import rmse
import matplotlib.pyplot as plt


class TestIpls(unittest.TestCase):
    def test_cmp_with_ss_impl(self):
        K = 50
        trajs, noise, ms_cmp, Ps_cmp = get_specific_states_from_file(Path.cwd() / "data/ipls_paper")
        ms_cmp = ms_cmp.reshape((K, 1))
        Ps_cmp = Ps_cmp.reshape((K, 1, 1))

        motion_model = NonStationaryGrowth(alpha=0.9, beta=10, gamma=8, delta=1.2, proc_noise=1)
        meas_model = Cubic(coeff=1 / 20, meas_noise=1)
        meas_model = Quadratic(coeff=1 / 20, meas_noise=1)

        # The paper simply states that "[t]he SLRs have been implemented using the unscented transform
        # with N s = 2n x + 1 sigma-points and the weight of the sigma-point located on the mean is 1/3."

        # The following settings ensures that:
        # a) w_0 (the weight of the mean sigma point) is 1/3
        # b) there is no difference for the weights for the mean and cov estimation
        sigma_point_method = UnscentedTransform(1, 0, 1 / 2)

        traj = trajs[:, 0].reshape((K, 1))
        meas = gen_measurements(traj, noise[:, 0], meas_model)
        prior_mean = np.atleast_1d(5)
        prior_cov = np.atleast_2d([4])

        num_iter = 1
        ipls = SigmaPointIpls(motion_model, meas_model, sigma_point_method, num_iter)
        _, _, ms_ipls, Ps_ipls, _ = ipls.filter_and_smooth(meas, prior_mean, prior_cov, cost_fn=None)

        self.assertTrue(np.allclose(ms_ipls, ms_cmp))
        self.assertTrue(np.allclose(Ps_ipls, Ps_cmp))
