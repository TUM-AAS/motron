from typing import Tuple, Union

import torch

from motion.components.quaternion_mvn import QuaternionMultivariateNormal
from motion.components.quaternion_mvn_time_series import QuaternionMultivariateNormalTimeSeries
from motion.components.to_gaussian import ToGaussian
from motion.skeleton import Skeleton
from motion.bingham import BinghamMixtureModel
from motion.quaternion import Quaternion
from motion.core.mgs2s.mgs2s import MGS2S


class Motion(torch.nn.Module):
    def __init__(self,
                 skeleton: Skeleton,
                 T: torch.Tensor = None,
                 G: Union[torch.Tensor, torch.nn.Parameter] = None,
                 **kwargs):
        super(Motion, self).__init__()

        self.param_groups = [{
            'teacher_forcing_factor': 0.,
            'z_dropout': 0.,
            'kl_weight': 0.
        }]

        num_nodes = skeleton.num_joints

        # Core Model
        self.core = MGS2S(num_nodes=num_nodes,
                          input_size=8,
                          G=G,
                          T=T,
                          param_groups=self.param_groups,
                          **kwargs
                          )

        self.to_gaussian = ToGaussian()

        #self.node_dropout = NodeDropout(0.2)

    def forward(self, x: torch.Tensor, ph: int = 1, y: torch.Tensor = None) \
            -> Tuple[torch.distributions.Distribution, dict]:
        if not self.training:
            y = None

        #x = self.node_dropout(x)

        # Calculate q_dot and concat it to q as input
        q_dot = Quaternion.mul_(x[:, 1:], Quaternion.conjugate_(x[:, :-1]))
        q_dot = torch.cat([torch.zeros_like(q_dot[:, [0]]), q_dot], dim=1)

        #x_scaled = x[..., 1:] / ((1+1e-1) - x[..., [0]])
        #q_dot_scaled = q_dot[..., 1:] / ((1+1e-1) - q_dot[..., [0]])
        #x_scaled = torch.cat([x_scaled, q_dot_scaled], dim=-1).contiguous()
        x_x_dot = torch.cat([x, q_dot], dim=-1).contiguous()

        q, Z_raw, z, q0, kwargs = self.core(x_x_dot, x, y, ph)

        # Permute from [B, Z, T, N, D] to [B, T, N, Z, D]
        q = q.permute(0, 2, 3, 1, 4).contiguous()
        Z_raw = Z_raw.permute(0, 2, 3, 1, 4).contiguous()

        # p_bmm = BinghamMixtureModel(z, LieMultivariateNormal(loc=q, scale_tril=scale_tril))

        log_std, correlation = Z_raw.split(3, dim=-1)

        p_dq = QuaternionMultivariateNormal(qloc=q, std=torch.exp(log_std), correlation=0.99*torch.tanh(correlation))

        q0 = QuaternionMultivariateNormal(qloc=q0, std=1e-2 * torch.ones_like(log_std[:, 0]),
                                          correlation=torch.zeros_like(correlation[:, 0]))
        p_bmm = BinghamMixtureModel(z, QuaternionMultivariateNormalTimeSeries(q0=q0, dq=p_dq))

        return p_bmm, {**kwargs}

    def loss(self, y_pred, y, **kwargs):
        return self.core.loss(y_pred, y, **kwargs)

    def hparams(self) -> dict:
        return {}