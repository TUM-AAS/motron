from functools import cached_property

import numpy as np
import torch

from motion.bingham.lookup_table import LookupTable


class Normalized(torch.distributions.constraints.Constraint):
    def check(self, value):
        return (torch.abs(torch.norm(value, dim=-1) - 1) < 1e-6).all()


class Bingham(torch.distributions.Distribution):
    arg_constraints = {'concentration': torch.distributions.constraints.real}
    support = Normalized

    def __init__(self, M: torch.Tensor, Z: torch.Tensor):
        batch_shape, event_shape = M.shape[:-2], M.shape[-1:]
        dim = event_shape[-1]
        # Check Dimensions
        assert M.shape[-1] * M.shape[-2] == dim**2, 'M is not square'
        assert Z.shape[-1] == dim, 'Z has wrong number of rows'

        # Enforce last entry of Z to be zero
        assert (Z[..., -1] == 0.).all(),  'last entry of Z needs to be zero'

        # Enforce z1<=z2<=...<=z(d-1)<=0=z(d)
        assert ((Z[..., 1:] - Z[..., :-1]) >= 0.).all(), 'values in Z have to be ascending'

        assert (Z >= -2000.).all(), 'Lookup table is only calculated until 900'

        # enforce that M is orthogonal
        assert ((M @ M.transpose(-2, -1) - torch.eye(dim, device=M.device)) < 1e-4).all(), 'M is not orthogonal'

        self.M = M
        self.Z = Z

        self.lookup_table = LookupTable()
        self.lookup_table_range = self.lookup_table['range'].to(M.device)
        self.lookup_table_F = self.lookup_table[f"F_{event_shape[0] - 1}d"].to(M.device)
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    @cached_property
    def F(self):
        Z = self.Z

        table_range = self.lookup_table_range
        n = table_range.shape[0]
        table_range_diff = table_range[1:] - table_range[:-1]

        y = torch.sqrt(-Z[..., :-1])

        idx = torch.searchsorted(table_range, y).clip(0, n - 2)

        range_tensor = table_range[idx]
        d_tensor = table_range_diff[idx]

        i0 = idx[..., 0]
        j0 = idx[..., 1]
        k0 = idx[..., 2]

        F000 = self.lookup_table_F[i0, j0, k0]
        F001 = self.lookup_table_F[i0, j0, k0 + 1]
        F010 = self.lookup_table_F[i0, j0 + 1, k0]
        F011 = self.lookup_table_F[i0, j0 + 1, k0 + 1]
        F100 = self.lookup_table_F[i0 + 1, j0, k0]
        F101 = self.lookup_table_F[i0 + 1, j0, k0 + 1]
        F110 = self.lookup_table_F[i0 + 1, j0 + 1, k0]
        F111 = self.lookup_table_F[i0 + 1, j0 + 1, k0 + 1]

        y = (torch.sqrt(-Z[..., :-1]) - range_tensor) / d_tensor
        y0 = y[..., 0]
        y1 = y[..., 1]
        y2 = y[..., 2]

        F00 = F000 + y2 * (F001 - F000)
        F01 = F010 + y2 * (F011 - F010)
        F10 = F100 + y2 * (F101 - F100)
        F11 = F110 + y2 * (F111 - F110)

        F0 = F00 + y1 * (F01 - F00)
        F1 = F10 + y1 * (F11 - F10)

        return F0 + y0 * (F1 - F0)

    @cached_property
    def C(self):
        return self.M.transpose(-2, -1) @ torch.diag_embed(self.Z) @ self.M

    def log_prob(self, value):
        l_p = torch.sum(value * (self.C @ value.unsqueeze(-1)).squeeze(-1), dim=-1) - torch.log(self.F)
        return l_p

    def sample(self, sample_shape=torch.Size(), burn_in=5, sample_rate=10):
        """
        Generate samples from Bingham distribution
        based on Glover's implementation in libBingham
        uses Metropolis-Hastings
        see http://en.wikipedia.org/wiki/Metropolis-Hastings_algorithm
        :param sample_rate:
        :param burn_in:
        :param sample_shape:
        :return:
        """
        bs = self.Z.shape[:-1]
        n = int(np.prod(list(sample_shape)))
        d = self.event_shape[-1]

        x = self.mean
        z = torch.sqrt(-1. / (self.Z - 1))

        target = self.log_prob(x)  # target
        proposal = AngularCentralGaussian(self.M, z).log_prob(x)  # proposal

        x2 = ((torch.randn((n * sample_rate + burn_in,) + bs + (d,)) * z).unsqueeze(-2) @ self.M).squeeze(-2)  # sample Gaussian
        x2 = x2 / x2.norm(dim=-1, keepdim=True)  # normalize

        targets = self.log_prob(x2)
        proposals = AngularCentralGaussian(self.M, z).log_prob(x2)
        states = []

        # Random walk
        for i in range(n * sample_rate + burn_in):
            acceptance = torch.exp(targets[i] - target + proposal - proposals[i])  # log space
            mask = (acceptance > torch.rand_like(acceptance)).type_as(acceptance)
            x = mask.unsqueeze(-1) * x2[i] + (1 - mask.unsqueeze(-1)) * x
            proposal = mask * proposals[i] + (1 - mask) * proposal
            target = mask * targets[i] + (1 - mask) * target
            states.append(x)

        sampled_states = torch.stack(states, dim=0)[burn_in + 1::sample_rate]
        return sampled_states.view(sample_shape + sampled_states.shape[1:])

    def __mul__(self, other):
        C = self.C + other.C  # new exponent

        C = 0.5 * (C + C.transpose(-2, -1))  # Ensure symmetry of C, asymmetry may arise as a consequence of a numerical instability earlier.

        D, V = torch.symeig(C, eigenvectors=True)  # Eigenvalue decomposition
        Z = D
        Z = Z - Z[..., -1]  # last entry should be zero
        M = V
        return Bingham(M , Z)

    @property
    def mode(self):
        return self.M[..., -1, :]

    @property
    def mean(self):
        return self.mode

    @classmethod
    def zero_rot(cls, Z):
        M = torch.diag_embed(torch.ones_like(Z), offset=1)
        M[..., -1, 0] = 1.
        Z = torch.cat([Z, torch.zeros(Z.shape[:-1] + (1,), device=Z.device)], dim=-1)
        return cls(M, Z)


class AngularCentralGaussian(torch.distributions.Distribution):
    def __init__(self, M: torch.Tensor, Z: torch.Tensor):
        batch_shape, event_shape = M.shape[:-2], M.shape[-1:]
        dim = event_shape[-1]
        # Check Dimensions
        assert M.shape[-1] * M.shape[-2] == dim ** 2, 'M is not square'
        assert Z.shape[-1] == dim, 'Z has wrong number of rows'

        # enforce that M is orthogonal
        assert ((M @ M.transpose(-2, -1) - torch.eye(dim, device=M.device)) < 1e-4).all(), 'M is not orthogonal'

        self.M = M
        self.Z = Z
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def log_prob(self, value):
        d = self.event_shape[-1]
        S_inv = self.M.transpose(-2, -1) @ torch.diag_embed(1. / (torch.square(self.Z))) @ self.M
        P = 1 / (torch.prod(self.Z, dim=-1) * self.unit_sphere_surface)
        md = torch.sum((value.unsqueeze(-2) @ S_inv) * value.unsqueeze(-2), dim=-1).squeeze(-1)  # mahalanobis distance
        P = P * torch.pow(md, -d / 2)
        return torch.log(P)

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


# from motion.components.to_bingham import ToBingham
#
# q = torch.rand((1, 4))
# #q[..., 0] = 1.
# q = torch.nn.functional.normalize(q, dim=-1)
# #print(q[0])
# Z = torch.ones((1, 3)) * 500
#
#
# M, Z = ToBingham(1999)(q, Z)
#
# b = Bingham(M, Z)
#
# a = b.log_prob(q)
#
# print(a)
# c = q.clone()
# c[..., 1:] = - c[..., 1:]
# print(b.log_prob(c))
