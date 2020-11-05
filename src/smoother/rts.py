"""Rauch-Tung-Striebel Smoother"""
from src.smoother.base import Smoother


class RtsSmoother(Smoother):
    def __init__(self, motion_model):
        self.motion_model = motion_model

    def _motion_lin(self, state, cov):
        return self.motion_model
