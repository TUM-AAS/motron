from typing import List

import torch
from torch.distributions.utils import lazy_property

from motion.distribution import MultivariateNormal, QuaternionMultivariateNormal
from motion.quaternion import Quaternion

tril_indices = torch.tril_indices(row=3, col=3, offset=-1)

class QuaternionMultivariateNormalTimeSeries(torch.distributions.Distribution):
    def __init__(self, p_q0: QuaternionMultivariateNormal, p_dq: QuaternionMultivariateNormal):
        self.p_q0 = p_q0
        self.p_dq = p_dq
        self.T = p_dq.batch_shape[1]
        batch_shape, event_shape = p_dq.batch_shape, p_dq.event_shape
        super().__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=False)

    @property
    def event_shape(self):
        return torch.Size((4, ))

    @lazy_property
    def integrated(self):
        rotation_mat = self.p_dq.qloc.rotation_matrix
        covariance_matrix = self.p_dq.covariance_matrix
        q_mat = self.p_dq.qloc._q_matrix()

        cov = self.p_q0.covariance_matrix
        q = self.p_q0.qloc.q

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

        correlation = cov[..., tril_indices[0], tril_indices[1]] / std_combinations_prod
        return QuaternionMultivariateNormal(qloc=qloc, std=std, correlation=correlation, fix_rho23=False)

    def log_prob(self, value):
        return self.integrated.log_prob(value)

    def rsample(self, sample_shape=torch.Size()):
        samples = []
        q = self.p_q0.sample(sample_shape)
        dq = self.p_dq.sample(sample_shape)
        for t in range(self.T):
            q = Quaternion.mul_(dq[:, :, t], q)
            samples.append(q)
        return torch.stack(samples, dim=2)

    @property
    def mean(self):
        return self.integrated.mean
