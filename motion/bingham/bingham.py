from functools import cached_property

import numpy as np
import torch

from .lookup_table import LookupTable


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

    def xi2CGFDeriv(self, t, dim, la, derriv):
        """
        First four derivatives of the cumulant generating function.
        :param dim:
        :param la:
        :param derriv:
        :return:
        """
        res = torch.zeros_like(la)
        for i in range(dim):
            if i == derriv:
                scale = 3.0
            else:
                scale = 1.0

            res[..., 0] += scale * 0.5/(la[i]-t)
            res[..., 1] += scale * 0.5/( (la[i]-t)*(la[i]-t))
            res[..., 2] += scale * 1/( (la[i]-t)*(la[i]-t)*(la[i]-t) )
            res[..., 3] += scale * 3/( (la[i]-t)*(la[i]-t)*(la[i]-t)*(la[i]-t) )

        return res

    # @property
    # def F_saddle(self):
    #     minEl = self.Z[..., 0]
    #     Z = self.Z - (minEl - 0.1)
    #     scaleFactor = torch.exp(-minEl + 0.1)
    #     minEl = 0.1
    #
    #     la = (double *)
    #     malloc(dim * sizeof(double));
    #     memcpy(la, z, dim * sizeof(double));
    #
    #     t = findRootNewton(dim, la, minEl);
    #
    #     xi2CGFDeriv(t, dim, la, hK, -1);
    #
    #     // T = 1 / 8 rho4 - 5 / 24 rho3 ^ 2
    #     T=1.0 / 8 * (hK[3] / (hK[1] * hK[1])) - 5.0 / 24 * (hK[2] * hK[2] / (hK[1] * hK[1] * hK[1]) );
    #
    #     result[0] = sqrt(2 * pow(M_PI, dim-1)) * exp(-t) / sqrt(hK[1]) * scaleFactor;
    #
    #     for (i=0; i < dim; i++) {
    #     result[0] /= sqrt(la[i]-t);
    #     }
    #     result[1] = result[0] * (1+T);
    #     result[2] = result[0] * exp(T);
    #
    #     }

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
        n = int(np.prod(list(sample_shape)))
        d = self.event_shape[-1]

        x = self.mode
        z = torch.sqrt(-1. / (self.Z - 1))

        target = self.log_prob(x)  # target
        proposal = AngularCentralGaussian(self.M, z).log_prob(x)  # proposal

        x2 = (torch.randn((n * sample_rate + burn_in, d)) * z) @ self.M  # sample Gaussian
        x2 = x2 / x2.norm(dim=-1, keepdim=True)  # normalize

        targets = self.log_prob(x2)
        proposals = AngularCentralGaussian(self.M, z).log_prob(x2)
        states = []

        # Random walk
        for i in range(n * sample_rate + burn_in):
            acceptance = torch.exp(targets[..., i] - target + proposal - proposals[..., i])  # log space
            mask = (acceptance > torch.rand_like(acceptance)).type_as(acceptance)
            x = mask * x2[..., i, :] + (1 - mask) * x
            proposal = mask * proposals[..., i] + (1 - mask) * proposal
            target = mask * targets[..., i] + (1 - mask) * target
            states.append(x)

        sampled_states = torch.stack(states, dim=1)[..., burn_in + 1::sample_rate, :]
        return sampled_states.view(sampled_states.shape[:-2] + sample_shape + sampled_states.shape[-1:])

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


def qmul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

# q = torch.tensor([[[1. , 2., 3. , 4.]]])
# q = q / q.norm()
#
# M = M_uns = torch.stack([
#             qmul(q, torch.tensor([0., 0, 0, 1.]).expand([1, 1] + [4])),
#             qmul(q, torch.tensor([0, 1., 0, 0]).expand([1, 1] + [4])),
#             qmul(q, torch.tensor([0, 0, 1., 0]).expand([1, 1] + [4])),
#             qmul(q, torch.tensor([1, 0, 0, 0.]).expand([1, 1] + [4])),
#         ], dim=-2)
#
# Z = torch.tensor([[[-900., -900, -900, 0]]])
#
# b = Bingham(M, Z)
#
# print(b.log_prob(qmul(q, torch.tensor([1., 0, 0.0, 0.]))))
#
# q2 = torch.tensor([[[1. , 2., 3. , 4.01]]])
# q2 = q2 / q.norm()
# print(b.log_prob(q2))