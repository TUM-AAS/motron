import torch
import numpy as np
from torch.distributions import MultivariateNormal, MixtureSameFamily


class GaussianMixtureModel(MixtureSameFamily):
    def __init__(self, mix, loc, scale_tril):
        self.mix = mix
        self.loc = loc
        self.scale_tril = scale_tril
        super().__init__(mix, MultivariateNormal(loc=loc, scale_tril=scale_tril))

    @classmethod
    def from_vector_params(cls, mix, loc, log_diag, scale_tril_v):
        diag = torch.exp(log_diag)
        diag_m = torch.diag_embed(diag)
        tril_m = torch.zeros_like(diag_m)
        tril_indices = torch.tril_indices(row=3, col=3, offset=-1)
        tril_m[..., tril_indices[0], tril_indices[1]] = scale_tril_v
        scale_tril = diag_m + tril_m
        return cls(mix, loc, scale_tril)

    def __getitem__(self, item):
        new_mix = torch.distributions.Categorical(logits=self.mixture_distribution.logits[item])
        return MixtureSameFamily(new_mix, MultivariateNormal(self.loc[item], scale_tril=self.L[item]))

    @property
    def mode(self):
        probs = self._pad_mixture_dimensions(
            torch.nn.functional.one_hot(
                self.mixture_distribution.probs.argmax(dim=-1), num_classes=self.mixture_distribution.probs.shape[-1]
            )
        )
        return torch.sum(probs * self.component_distribution.mean,
                         dim=-1 - self._event_ndims)  # [B, E]


class MultivariateNormal2(MultivariateNormal):
    def __init__(self, mus, log_sigmas, corrs):
        corrs = torch.tanh(corrs)
        self.dimensions = 3
        self.device = log_sigmas.device
        self.mus = self.reshape_to_components(mus)  # [..., N, 3]
        self.log_sigmas = self.reshape_to_components(log_sigmas)  # [..., N, 3]
        self.sigmas = torch.exp(self.log_sigmas)  # [..., N, 3]
        self.corrs = self.reshape_to_components(corrs)  # [..., N, 3] Order = [rho_xy, rho_xz, rho_yz]

        # self.L = torch.cholesky(self.get_covariance_matrix())
        self.one_minus_rhoxy2 = torch.clamp(1 - self.corrs[..., 0] ** 2, 1e-5, 1)  # [..., N]
        self.rhoyz_minus_rhoxy_rhoxz = self.corrs[..., 2] - self.corrs[..., 0] * self.corrs[..., 1]
        sqrt_term = torch.clamp(
            1.0 - (self.corrs[..., 1] ** 2) - (self.rhoyz_minus_rhoxy_rhoxz ** 2) / self.one_minus_rhoxy2, 1e-5, 1)

        bs_ts_fs = list(mus.shape[:-1])
        z = torch.zeros(bs_ts_fs, device=mus.device)
        self.L = torch.stack(
            [torch.stack([self.sigmas[..., 0], z, z], dim=-1),
             torch.stack(
                 [self.sigmas[..., 1] * self.corrs[..., 0], self.sigmas[..., 1] * torch.sqrt(self.one_minus_rhoxy2),
                  z], dim=-1),
             torch.stack([self.sigmas[..., 2] * self.corrs[..., 1],
                          self.sigmas[..., 2] * self.rhoyz_minus_rhoxy_rhoxz / torch.sqrt(self.one_minus_rhoxy2), \
                          self.sigmas[..., 2] * torch.sqrt(sqrt_term)], dim=-1)]
            , dim=-2)

        super(MultivariateNormal2, self).__init__(loc=mus, scale_tril=self.L)

    @classmethod
    def from_log_pis_mus_cov_mats(cls, log_pis, mus, cov_mats):
        sigmas = torch.clamp(torch.stack(
            [torch.sqrt(cov_mats[..., 0, 0]), torch.sqrt(cov_mats[..., 1, 1]), torch.sqrt(cov_mats[..., 2, 2])],
            dim=-1), min=1e-5)
        log_sigmas = torch.log(sigmas)

        rho_xy = cov_mats[..., 0, 1] / torch.clamp(torch.sqrt(cov_mats[..., 0, 0] * cov_mats[..., 1, 1]), min=1e-5)
        rho_xz = cov_mats[..., 0, 2] / torch.clamp(torch.sqrt(cov_mats[..., 0, 0] * cov_mats[..., 2, 2]), min=1e-5)
        rho_yz = cov_mats[..., 1, 2] / torch.clamp(torch.sqrt(cov_mats[..., 1, 1] * cov_mats[..., 2, 2]), min=1e-5)

        corrs = torch.stack([rho_xy, rho_xz, rho_yz], dim=-1)

        return cls(log_pis, mus, log_sigmas, corrs)

    def log_prob(self, value):
        r"""
        Calculates the log probability of a value using the PDF for bivariate normal distributions:

        .. math::
            f(x | \mu, \sigma, \rho)={\frac {1}{2\pi \sigma _{x}\sigma _{y}{\sqrt {1-\rho ^{2}}}}}\exp
            \left(-{\frac {1}{2(1-\rho ^{2})}}\left[{\frac {(x-\mu _{x})^{2}}{\sigma _{x}^{2}}}+
            {\frac {(y-\mu _{y})^{2}}{\sigma _{y}^{2}}}-{\frac {2\rho (x-\mu _{x})(y-\mu _{y})}
            {\sigma _{x}\sigma _{y}}}\right]\right)

        :param value: The log probability density function is evaluated at those values.
        :return: Log probability
        """
        # x: [..., 3]
        #value = torch.unsqueeze(value, dim=-2)  # [..., 1, 3]
        dx = value - self.mus  # [..., N, 3]

        # exp_nominator = ((torch.sum((dx/self.sigmas)**2, dim=-1)  # first and second term of exp nominator
        #                   - 2*self.corrs*torch.prod(dx, dim=-1)/torch.prod(self.sigmas, dim=-1)))    # [..., N]

        # component_log_p = -(2*np.log(2*np.pi)
        #                     + torch.log(self.one_minus_rho2)
        #                     + 2*torch.sum(self.log_sigmas, dim=-1)
        #                     + exp_nominator/self.one_minus_rho2) / 2

        # Torch method
        # print(torch.det(sigma).view(-1).max(),torch.det(sigma).view(-1).min())
        sigma = self.get_covariance_matrix()
        sigma_inv = torch.inverse(sigma)
        exponent = -(dx.unsqueeze(-2).matmul(sigma_inv).matmul(dx.unsqueeze(-1))).squeeze(dim=-1).squeeze(dim=-1) / 2

        component_log_p = exponent - 3 * np.log(2 * np.pi) - torch.log(torch.clamp(torch.det(sigma), min=1e-5))
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # print(component_log_p.view(-1).abs().max())
        # print('============================')
        # print(exponent.view(-1).abs().max())
        # print('----------------------------')
        # print(torch.det(sigma).view(-1).abs().min())
        # print((3*np.log(2*np.pi)))
        # # Polynomial method
        # x_0 = dx[...,0]/self.sigmas[...,0]
        # y_0 = dx[...,1]/self.sigmas[...,1]
        # z_0 = dx[...,2]/self.sigmas[...,2]

        # xbar = self.corrs[...,2]
        # ybar = self.corrs[...,1]
        # zbar = self.corrs[...,0]

        # term1 = (x_0*xbar + y_0*ybar + z_0*zbar)**2
        # term2 = x_0**2*(1-2*xbar**2) + y_0**2*(1-2*ybar**2) + z_0**2*(1-2*zbar**2)
        # term3 = -2*(x_0*z_0*ybar + x_0*y_0*zbar + y_0*z_0*xbar)
        # gamma = 1 -xbar**2 -ybar**2 -zbar**2 +2*xbar*ybar*zbar
        # exponent = -(term1 + term2 + term3)/(2*gamma)
        # det = gamma * torch.prod(self.sigmas, dim=-1)**2

        # component_log_p = exponent - 3*np.log(2*np.pi) - torch.log(det)
        # print(component_log_p.shape)

        # print(component_log_p.view(-1).max())
        # print(component_log_p.view(-1).min())
        # print(torch.logsumexp(self.log_pis + component_log_p, dim=-1).shape)
        return component_log_p

    def mode(self):
        """
        Calculates the mode of the GMM by calculating probabilities of a 2D mesh grid

        :param required_accuracy: Accuracy of the meshgrid
        :return: Mode of the GMM
        """
        assert self.mus.shape[-2] == 1  # TODO Old methode for more than one component
        return torch.squeeze(self.mus, dim=-2)

    def reshape_to_components(self, tensor):
        if len(tensor.shape) == 5:
            return tensor
        return torch.reshape(tensor, list(tensor.shape[:-1]) + [self.components, self.dimensions])

    def get_covariance_matrix(self):
        # E = self.L.transpose(-2,-1).matmul(self.L)

        cov_xy = self.corrs[..., 0] * self.sigmas[..., 0] * self.sigmas[..., 1]
        cov_xz = self.corrs[..., 1] * self.sigmas[..., 0] * self.sigmas[..., 2]
        cov_yz = self.corrs[..., 2] * self.sigmas[..., 1] * self.sigmas[..., 2]

        E = torch.stack([torch.stack([self.sigmas[..., 0] ** 2, cov_xy, cov_xz], dim=-1),
                         torch.stack([cov_xy, self.sigmas[..., 1] ** 2, cov_yz], dim=-1),
                         torch.stack([cov_xz, cov_yz, self.sigmas[..., 2] ** 2], dim=-1)],
                        dim=-2)

        return E