import torch
import numpy as np

class ACG(torch.distributions.Distribution):
    def __init__(self, M, Z):
        batch_shape, event_shape = M.shape[:-2], M.shape[-1:]
        self.M = M  # cov eigenvectors
        self.Z = Z  # cov eigenvalues
        self.A = M.transpose(-2, -1) @ torch.diag_embed(Z) @ M
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def log_prob(self, value):
        d = self.event_shape[-1]
        A_inv = self.M.transpose(-2, -1) @ torch.diag_embed(1. / self.Z) @ self.M
        log_det_A = torch.sum(torch.log(self.Z), dim=-1)

        md = torch.sum((value.unsqueeze(-2) @ A_inv) * value.unsqueeze(-2), dim=-1).squeeze(-1)
        p = -0.5 * d * torch.log(md) - 0.5 * log_det_A - np.log(self.unit_sphere_surface)

        return p

    def sample(self, sample_shape=torch.Size()):
        s = torch.distributions.MultivariateNormal(loc=torch.zeros_like(self.Z),
                                                   scale_tril=self.M.transpose(-2, -1) @ torch.diag_embed(
                                                       torch.sqrt(self.Z))).sample(sample_shape)
        return s / s.square().sum(-1, keepdim=True).sqrt()

    def rsample(self, sample_shape=torch.Size()):
        s = torch.distributions.MultivariateNormal(loc=torch.zeros_like(self.Z),
                                                   scale_tril=self.M.transpose(-2, -1) @ torch.diag_embed(
                                                       torch.sqrt(self.Z))).rsample(sample_shape)
        return s / s.square().sum(-1, keepdim=True).sqrt()

    @property
    def variance(self):
        return torch.distributions.MultivariateNormal(loc=torch.zeros_like(self.Z),
                                                      scale_tril=self.M.transpose(-2, -1) @ torch.diag_embed(
                                                          torch.sqrt(self.Z))).variance

    @property
    def mode(self):
        d = self.event_shape[-1]
        _, eigv_imax = torch.max(self.Z, dim=-1)
        m = self.M.gather(dim=-2,
                          index=eigv_imax.unsqueeze(-1).unsqueeze(-1).repeat(len(eigv_imax.shape) * [1] + [1, d]))
        return m.squeeze(-2)

    @property
    def mean(self):
        return self.mode

    @property
    def unit_sphere_surface(self):
        dim = self.event_shape[-1]
        if dim == 2:
            return 2 * np.pi
        elif dim == 3:
            return 4 * np.pi
        elif dim == 4:
            return 2 * np.pi ** 2
        else:
            raise NotImplementedError