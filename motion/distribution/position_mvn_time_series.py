import torch
from torch.distributions.utils import lazy_property

from motion.distribution import MultivariateNormal

tril_indices = torch.tril_indices(row=3, col=3, offset=-1)


class PositionMultivariateNormalTimeSeries(torch.distributions.Distribution):
    def __init__(self, p_p0: MultivariateNormal, p_dp: MultivariateNormal):
        self.p_p0 = p_p0
        self.p_dp = p_dp
        self.T = p_dp.batch_shape[1]
        batch_shape, event_shape = p_dp.batch_shape, p_dp.event_shape
        super().__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=False)

    @lazy_property
    def integrated(self):
        cov = self.p_p0.covariance_matrix.unsqueeze(dim=1) + torch.cumsum(self.p_dp.covariance_matrix, dim=1)
        p = self.p_p0.mean.unsqueeze(dim=1) + torch.cumsum(self.p_dp.mean, dim=1)

        std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))
        std_combinations_prod = torch.stack([std[..., 0] * std[..., 1],
                                             std[..., 0] * std[..., 2],
                                             std[..., 1] * std[..., 2]], dim=-1)

        correlation = cov[..., tril_indices[0], tril_indices[1]] / std_combinations_prod
        return MultivariateNormal(loc=p, std=std, correlation=correlation, fix_rho23=False)

    def log_prob(self, value):
        return self.integrated.log_prob(value)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        samples = []
        p = self.p_p0.sample(sample_shape)
        dp = self.p_dp.sample(sample_shape)
        for t in range(self.T):
            p = p + dp[:, :, t]
            samples.append(p)
        return torch.stack(samples, dim=2)

    @property
    def mean(self):
        return self.integrated.mean
