from typing import Tuple, Union

import torch

from motion.bingham.bingham import AngularCentralGaussian
from motion.components.node_dropout import NodeDropout
from motion.skeleton import Skeleton
from motion.bingham import Bingham, BinghamMixtureModel
from motion.quaternion import Quaternion
from motion.components.to_bingham import ToBingham
from motion.core.mgs2s.mgs2s import MGS2S
from motion.van_mises_fisher import VonMisesFisher
from motion.van_mises_fisher.distributions.vmf_mm import VonMisesFisherMixtureModel


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

        self.to_bingham = ToBingham(4000.)

        self.node_dropout = NodeDropout(0.0)

    def forward(self, x: torch.Tensor, ph: int = 1, y: torch.Tensor = None) \
            -> Tuple[torch.distributions.Distribution, dict]:
        if not self.training:
            y = None

        # Calculate q_dot and concat it to q as input
        q_dot = Quaternion.mul_(x[:, 1:], Quaternion.conjugate_(x[:, :-1]))
        q_dot = torch.cat([torch.zeros_like(q_dot[:, [0]]), q_dot], dim=1)

        #x_scaled = x[..., 1:] / ((1+1e-1) - x[..., [0]])
        #q_dot_scaled = q_dot[..., 1:] / ((1+1e-1) - q_dot[..., [0]])
        #x_scaled = torch.cat([x_scaled, q_dot_scaled], dim=-1).contiguous()
        x_x_dot = torch.cat([x, q_dot], dim=-1).contiguous()

        q, Z_raw, z, kwargs = self.core(x_x_dot, x, y, ph)

        # Permute from [B, Z, T, N, D] to [B, T, N, Z, D]
        q = q.permute(0, 2, 3, 1, 4).contiguous()
        Z_raw = Z_raw.permute(0, 2, 3, 1, 4).contiguous()

        M, Z = self.to_bingham(q, Z_raw)

        p_bmm = BinghamMixtureModel(z, Bingham(M, Z))
        #p_bmm = VonMisesFisherMixtureModel(z, VonMisesFisher(q, torch.exp(Z_raw)))

        return p_bmm, {**kwargs}

    def loss(self, y_pred, y, **kwargs):
        return self.core.loss(y_pred, y, **kwargs)

    def hparams(self) -> dict:
        return {}