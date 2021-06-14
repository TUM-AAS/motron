from typing import Tuple, Union, Optional

import torch

from motion.components.node_dropout import NodeDropout
from motion.distribution import (QuaternionMultivariateNormal, QuaternionMultivariateNormalTimeSeries,
                                 TimeSeriesMixtureModel, PositionMultivariateNormalTimeSeries, MultivariateNormal)

from motion.skeleton import Skeleton
from motion.quaternion import Quaternion
from motion.core.mgs2s.mgs2s import MGS2S


class Motion(torch.nn.Module):
    def __init__(self,
                 skeleton: Skeleton,
                 latent_size: int,
                 T: torch.Tensor = None,
                 G: Union[torch.Tensor, torch.nn.Parameter] = None,
                 node_dropout: bool = 0.,
                 position: bool = False,
                 ignore_absolute_root_rotation: bool = False,
                 **kwargs):

        super(Motion, self).__init__()

        self.param_groups = [{
            'teacher_forcing_factor': 0.,
        }]

        self._latent_size = latent_size
        self._position = position

        self.ignore_absolute_root_rotation = ignore_absolute_root_rotation

        if position:
            # Additional node (and node type) for position
            T = torch.cat([T, (T.max() + 1).unsqueeze(0)], dim=-1)

        num_nodes = skeleton.num_joints

        # Core Model
        self.core = MGS2S(num_nodes=num_nodes + (1 if position else 0),
                          input_size=8,
                          G=G,
                          T=T,
                          param_groups=self.param_groups,
                          latent_size=latent_size,
                          position=position,
                          **kwargs
                          )

        if node_dropout > 0.:
            self.node_dropout = NodeDropout(node_dropout)
        else:
            self.node_dropout = None

    def forward(self, q: torch.Tensor, p: Optional[torch.tensor] = None,  ph: int = 1,
                state=None, y: torch.Tensor = None) \
            -> Tuple[torch.distributions.Distribution, torch.distributions.Distribution, Tuple, dict]:
        if not self.training:
            y = None

        if self.node_dropout is not None:
            q = self.node_dropout(q)

        q0_state = q[:, [-1]].clone()
        if self._position:
            p0_state = p[:, [-1]].clone()
        else:
            p0_state = None

        # Calculate dq
        if state is None:
            dq = Quaternion.mul_(q[:, 1:], Quaternion.conjugate_(q[:, :-1]))
            dq = torch.cat([torch.zeros_like(dq[:, [0]]), dq], dim=1)

            dp = None
            if self._position:
                dp = 20 * (p[:, 1:] - p[:, :-1])
                dp = torch.cat([torch.zeros_like(dp[:, [0]]), dp], dim=1)

            core_state = None
        else:
            core_state, q_tm1, p_tm1 = state
            dq = Quaternion.mul_(q, Quaternion.conjugate_(q_tm1))

            dp = None
            if self._position:
                dp = 20 * (p - p_tm1)

            core_state = state[0]

        q0 = q[:, -1].clone().unsqueeze(-2).repeat_interleave(repeats=self._latent_size, dim=-2)

        if self.ignore_absolute_root_rotation:
            q[:, :, 0] = torch.zeros_like(q[:, :, 0])

        dq, dq_cov_lat, dp, dp_cov_lat, z_logits, core_state, kwargs = self.core(q, dq, p, dp, y, ph, core_state)

        # Permute from [B, Z, T, N, D] to [B, T, N, Z, D]
        dq = dq.permute(0, 2, 3, 1, 4).contiguous()
        dq_cov_lat = dq_cov_lat.permute(0, 2, 3, 1, 4).contiguous()

        log_std_q, correlation_q = dq_cov_lat.split(3, dim=-1)

        p_dq = QuaternionMultivariateNormal(qloc=dq, std=torch.exp(log_std_q),
                                            correlation=0.99*torch.tanh(correlation_q))

        p_q0 = QuaternionMultivariateNormal(qloc=q0, std=1e-2 * torch.ones_like(log_std_q[:, 0]),  # Measurement noise
                                            correlation=torch.zeros_like(correlation_q[:, 0]))

        p_q = TimeSeriesMixtureModel(mixture_distribution=torch.distributions.Categorical(logits=z_logits
                                                                                          .unsqueeze(1)
                                                                                          .unsqueeze(1)),
                                     component_distribution=QuaternionMultivariateNormalTimeSeries(p_q0=p_q0,
                                                                                                   p_dq=p_dq))
        if self._position:
            p0 = p[:, -1].unsqueeze(-2).repeat_interleave(repeats=self._latent_size, dim=-2)

            dp = dp.permute(0, 2, 1, 3).contiguous()
            dp_cov_lat = dp_cov_lat.permute(0, 2, 1, 3).contiguous()

            log_std_p, correlation_p = dp_cov_lat.split(3, dim=-1)

            p_dp = MultivariateNormal(loc=dp, std=torch.exp(log_std_p),
                                      correlation=0.99 * torch.tanh(correlation_p))

            p_p0 = MultivariateNormal(loc=p0, std=1e-2 * torch.ones_like(log_std_p[:, 0]),
                                      correlation=torch.zeros_like(correlation_p[:, 0]))

            p_p = TimeSeriesMixtureModel(mixture_distribution=torch.distributions.Categorical(logits=z_logits.unsqueeze(1)),
                                         component_distribution=PositionMultivariateNormalTimeSeries(p_p0=p_p0,
                                                                                                     p_dp=p_dp))
        else:
            p_p = None

        state = (core_state, q0_state, p0_state)
        return p_q, p_p, state, {**kwargs}

    def loss(self, y_pred, y, **kwargs):
        return self.core.loss(y_pred, y, **kwargs)

    def hparams(self) -> dict:
        return {}