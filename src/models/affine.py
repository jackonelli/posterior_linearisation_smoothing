"""Stochastic affine model for comparison with analytical KF"""
import numpy as np
from scipy.stats import multivariate_normal as mvn
from src.models.base import MotionModel, MeasModel


class AffineModel(MotionModel, MeasModel):
    def __init__(self, linear_map, offset, noise):
        """Affine model

        works as both measurement and motion model

        y = A * x + b + sigma, sigma ~ N(0, Sigma)
        linear_map = A,
        offset = b.
        noise = Sigma

        Maps the vector x in R^n to y in R^m

        linear_map (np.ndarray): Matrix R^(m, n)
        offset (np.ndarray): vector R^(m)
        """
        self.linear_map = linear_map
        self.noise = noise
        self.offset = offset

    def mapping(self, state: np.ndarray) -> np.ndarray:
        return self.linear_map @ state + self.offset

    def proc_noise(self, _time_step):
        return self.noise

    def meas_noise(self, _time_step):
        return self.noise

    def sample(self, x_sample):
        new_mean = (self.linear_map @ x_sample.T).T + self.offset
        num_samples, meas_dim = new_mean.shape
        if self.proc_noise.size == 1:
            proc_noise = self.proc_noise[0]
        else:
            proc_noise = self.proc_noise
        noise = mvn.rvs(mean=np.zeros((meas_dim,)), cov=proc_noise, size=x_sample.shape[0]).reshape(
            (num_samples, meas_dim)
        )

        return new_mean + noise
