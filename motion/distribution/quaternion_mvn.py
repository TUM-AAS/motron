import torch

from motion.distribution.multivariate_normal import MultivariateNormal
from motion.quaternion import Quaternion


class QuaternionMultivariateNormal(MultivariateNormal):
    def __init__(self, qloc, std=None, correlation=None, scale_tril=None, fix_rho23=True):
        self.qloc = Quaternion(qloc)
        super().__init__(loc=torch.zeros_like(qloc[..., :3]), std=std, correlation=correlation,
                         scale_tril=scale_tril, fix_rho23=fix_rho23)

    def __getitem__(self, item):
        return self.__class__(qloc=self.qloc.q[item],
                              std=self.std[item],
                              correlation=self.correlation[item],
                              scale_tril=self.scale_tril[item])

    def log_prob(self, value):
        qv = Quaternion(value)
        dist = (qv * self.qloc.conjugate).axis_angle
        lp = super().log_prob(dist)
        return lp

    def rsample(self, sample_shape=torch.Size()):
        s = super().rsample(sample_shape)
        s_quat = Quaternion(axis=s, angle=s.square().sum(-1).sqrt())
        r = (s_quat * self.qloc).q
        return r

    @property
    def mean(self):
        return self.qloc.q

    def __mul__(self, other):
        qloc = self.qloc * other.qloc
        rotation_mat = self.qloc.rotation_matrix
        rot_cov_other = rotation_mat @ other.covariance_matrix @ rotation_mat.transpose(-2, -1)
        cov = self.covariance_matrix + rot_cov_other

        std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))
        std_combinations_prod = torch.stack([std[..., 0] * std[..., 1],
                                             std[..., 0] * std[..., 2],
                                             std[..., 1] * std[..., 2]], dim=-1)

        tril_indices = torch.tril_indices(row=3, col=3, offset=-1)
        correlation = cov[..., tril_indices[0], tril_indices[1]] / std_combinations_prod

        return self.__class__(qloc=qloc.q, std=std, correlation=correlation, fix_rho23=False)
