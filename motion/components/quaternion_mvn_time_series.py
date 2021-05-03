from typing import List

import torch
from torch.distributions.utils import lazy_property

from motion.components.multivariate_normal import MultivariateNormal
from motion.components.quaternion_mvn import QuaternionMultivariateNormal
from motion.gaussian.mvn import LieMultivariateNormal
from motion.quaternion import Quaternion


class QuaternionMultivariateNormalTimeSeries(torch.distributions.Distribution):
    def __init__(self, q0: QuaternionMultivariateNormal, dq: QuaternionMultivariateNormal):
        self.q0 = q0
        self.dq = dq
        self.T = dq.batch_shape[1]
        batch_shape, event_shape = dq.batch_shape, dq.event_shape
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    @lazy_property
    def integrated(self):
        rotation_mat = self.dq.qloc.rotation_matrix
        covariance_matrix = self.dq.covariance_matrix
        q_mat = self.dq.qloc._q_matrix()

        cov = self.q0.covariance_matrix
        q = self.q0.qloc.q

        integrated_cov = list()
        integrated_q = list()
        for t in range(self.T):
            q = torch.matmul(q_mat[:, t], q.unsqueeze(-1)).squeeze(-1)
            rot_cov = rotation_mat[:, t] @ cov @ rotation_mat[:, t].transpose(-2, -1)
            cov = covariance_matrix[:, t] + rot_cov

            integrated_q.append(q)
            integrated_cov.append(cov)

        qloc = torch.stack(integrated_q, dim=1)
        cov = torch.stack(integrated_cov, dim=1)

        std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))
        std_combinations_prod = torch.stack([std[..., 0] * std[..., 1],
                                             std[..., 0] * std[..., 2],
                                             std[..., 1] * std[..., 2]], dim=-1)

        tril_indices = torch.tril_indices(row=3, col=3, offset=-1)
        correlation = cov[..., tril_indices[0], tril_indices[1]] / std_combinations_prod
        return QuaternionMultivariateNormal(qloc=qloc, std=std, correlation=correlation, fix_rho23=False)

    def log_prob(self, value):
        return self.integrated.log_prob(value)
        test = self.integrated.log_prob(value)
        log_prob = []
        q = self.q0
        for t in range(self.T):
            q = self.dq[:, t] * q
            log_prob.append(q.log_prob(value[:, t]))
        return torch.stack(log_prob, dim=1)
        q = torch.cat([self.q0.mean[..., 0, :].unsqueeze(1).unsqueeze(-2), value], dim=1)
        q_dot = Quaternion.mul_(q[:, 1:], Quaternion.conjugate_(q[:, :-1]))
        return self.dq.log_prob(q_dot)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        samples = []
        q = self.q0.sample(sample_shape)
        for t, dq_t in enumerate(self.dq):
            q = Quaternion.mul_(dq_t.sample(sample_shape), q)
            samples.append(q)
        return torch.stack(samples, dim=1)

    @property
    def mean(self):
        samples = []
        q = self.q0.mean
        dq = self.dq.mean
        for t in range(self.T):
            q = Quaternion.mul_(dq[:, t], q)
            samples.append(q)
        return torch.stack(samples, dim=1)

    @property
    def mean2(self):
        return self.dq.mean
