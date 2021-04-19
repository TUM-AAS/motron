from typing import List

import torch

from motion.components.multivariate_normal import MultivariateNormal
from motion.components.quaternion_mvn import QuaternionMultivariateNormal
from motion.quaternion import Quaternion


class QuaternionMultivariateNormalTimeSeries(torch.distributions.Distribution):
    def __init__(self, q0: QuaternionMultivariateNormal, q_ts: List[QuaternionMultivariateNormal]):
        self.q0 = q0
        self.q_ts = q_ts
        super().__init__()

    def log_prob(self, value):
        log_prob = []
        q = self.q0
        for t, dq in enumerate(self.q_ts):
            q = dq * q
            log_prob.append(q.log_prob(value[:, t]))
        return torch.stack(log_prob, dim=1)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        samples = []
        q = self.q0.sample(sample_shape)
        for t, dq in enumerate(self.q_ts):
            q = Quaternion.mul_(dq.sample(sample_shape), q)
            samples.append(q)
        return torch.stack(samples, dim=1)

    @property
    def mean(self):
        samples = []
        q = self.q0.mean
        for t, dq in enumerate(self.q_ts):
            q = Quaternion.mul_(dq.mean, q)
            samples.append(q)
        return torch.stack(samples, dim=1)
