import torch
from torch.distributions.utils import lazy_property

tril_indices = torch.tril_indices(row=3, col=3, offset=-1)
triu_indices = torch.triu_indices(row=3, col=3, offset=1)


class MultivariateNormal(torch.distributions.MultivariateNormal):
    def __init__(self, loc, std, correlation, scale_tril=None, fix_rho23=True):

        assert loc.shape[-1] == 3

        self.std = std
        self.correlation = correlation
        if scale_tril is None:
            rho12, rho13, rho23 = correlation[..., 0], correlation[..., 1], correlation[..., 2]
            rho12_square = torch.square(rho12)
            rho13_square = torch.square(rho13)
            if fix_rho23:
                squared_term = torch.sqrt(1. + rho12_square * rho13_square - rho12_square - rho13_square)
                rho23 = rho12 * rho13 + rho23 * squared_term
                self.correlation = torch.stack([rho12, rho13, rho23], dim=-1)
            zero = torch.zeros_like(rho12)
            one_minus_rho12_2 = 1 - rho12_square
            rho23_minus_rho12_rho_13 = rho23 - rho12 * rho13

            l00 = torch.ones_like(rho12)
            l10 = rho12
            l11 = torch.sqrt(one_minus_rho12_2)
            l20 = rho13
            l21 = rho23_minus_rho12_rho_13 / torch.sqrt(one_minus_rho12_2)
            l22 = torch.sqrt(1 - ((rho13_square + rho23 ** 2 - 2 * self.correlation.prod(dim=-1))
                                              / one_minus_rho12_2))

            assert not torch.isnan(l00).any()
            assert not torch.isnan(l10).any()
            assert not torch.isnan(l11).any()
            assert not torch.isnan(l20).any()
            assert not torch.isnan(l21).any()
            assert not torch.isnan(l22).any()

            scale_tril_correlation = torch.stack([torch.stack([l00, zero, zero], dim=-1),
                                                  torch.stack([l10, l11, zero], dim=-1),
                                                  torch.stack([l20, l21, l22], dim=-1)], dim=-2)

            scale_tril = torch.diag_embed(std) @ scale_tril_correlation
            assert not torch.isnan(scale_tril).any()
        super().__init__(loc=loc, scale_tril=scale_tril)

        @lazy_property
        def covariance_matrix(self):
            std_diag = torch.diag_embed(self.std)
            corr = torch.diag_embed(torch.ones_like(self.std))
            corr[..., tril_indices[0], tril_indices[1]] = self.correlation
            corr[..., triu_indices[0], triu_indices[1]] = self.correlation
            return std_diag @ corr @ std_diag.transpose(-2, -1)

    def __add__(self, other):
        std = torch.sqrt(self.std.square() + other.std.square())
        corr01 = ((self.correlation[..., 0] * self.std[..., 0] * self.std[..., 1] +
                   other.correlation[..., 0] * other.std[..., 0] * other.std[..., 1]) /
                  (std[..., 0] * std[..., 1]))
        corr02 = ((self.correlation[..., 1] * self.std[..., 0] * self.std[..., 2] +
                   other.correlation[..., 1] * other.std[..., 0] * other.std[..., 2]) /
                  (std[..., 0] * std[..., 2]))
        corr12 = ((self.correlation[..., 2] * self.std[..., 1] * self.std[..., 2] +
                   other.correlation[..., 2] * other.std[..., 1] * other.std[..., 2]) /
                  (std[..., 1] * std[..., 2]))
        correlation = torch.cat([corr01, corr02, corr12], dim=-1)
        return self.__class__(loc=self.loc + other.loc, std=std, correlation=correlation)


# C = torch.rand((1, 3, 3))
# C = (C @ C.transpose(-2, -1)) / 2
# C2 = torch.rand((1, 3, 3))
# C2 = (C2 @ C2.transpose(-2, -1)) / 2
# v = torch.rand((1, 3))
# #C = torch.diag_embed(v)
# loc = torch.zeros((1, 3))
# tril_indices = torch.tril_indices(row=3, col=3, offset=-1)
# corrs = C[..., tril_indices[0], tril_indices[1]]
# corrs[..., 0] /= torch.sqrt(C[..., 0, 0]) * torch.sqrt(C[..., 1, 1])
# corrs[..., 1] /= torch.sqrt(C[..., 0, 0]) * torch.sqrt(C[..., 2, 2])
# corrs[..., 2] /= torch.sqrt(C[..., 1, 1]) * torch.sqrt(C[..., 2, 2])
#
# corrs2 = C2[..., tril_indices[0], tril_indices[1]]
# corrs2[..., 0] /= torch.sqrt(C2[..., 0, 0]) * torch.sqrt(C2[..., 1, 1])
# corrs2[..., 1] /= torch.sqrt(C2[..., 0, 0]) * torch.sqrt(C2[..., 2, 2])
# corrs2[..., 2] /= torch.sqrt(C2[..., 1, 1]) * torch.sqrt(C2[..., 2, 2])
#
# mvn1 = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=C+C2)
#
#
# mvn2 = MultivariateNormal(loc=loc, std=torch.sqrt(torch.diagonal(C, dim1=-2, dim2=-1)), correlation=corrs)
# mvn3 = MultivariateNormal(loc=loc, std=torch.sqrt(torch.diagonal(C2, dim1=-2, dim2=-1)), correlation=corrs2)
# mvn4 = mvn2 + mvn3
#
# print(mvn1.covariance_matrix)
# print(mvn4.covariance_matrix)