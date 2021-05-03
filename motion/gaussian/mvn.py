import torch
import numpy as np

from motion.quaternion import Quaternion


class LieMultivariateNormal(torch.distributions.MultivariateNormal):
    def __init__(self, loc, scale_tril):
        self.loc_ = loc
        super().__init__(loc=torch.zeros_like(loc[..., :3]), scale_tril=scale_tril)

    def log_prob(self, value):
        v = Quaternion(value)
        l = Quaternion(self.loc_)
        d = (v * l.conjugate).axis_angle
        return super().log_prob(d)

    @property
    def mean(self):
        return self.loc_

    @property
    def mean2(self):
        return self.loc_
