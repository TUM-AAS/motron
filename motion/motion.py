from typing import Tuple, Union

import torch

from common.quaternion import inverse, qmul
from common.torch import Module
from directional_distributions import Bingham
from motion.core.mgs2s.mgs2s import MGS2S
from motion.distributions import BinghamMixtureModel, GaussianMixtureModel
from motion.distributions.chained_bingham import ChainedBinghamMixtureModel, ChainedBingham
from motion.dynamics import Linear


class Motion(Module):
    def __init__(self, graph_representation, **kwargs):
        super(Motion, self).__init__()

        self.param_groups = [{
            'teacher_forcing_factor': 0.
        }]

        self.graph_representation = graph_representation

        self.graph_influence_matrix = torch.nn.Parameter(self.graph_representation.adjacency_matrix)

        self._chain_list = self.graph_representation.chain_list

        # Core Model
        self.core = MGS2S(nodes=graph_representation.num_nodes,
                          graph_influence=self.graph_influence_matrix,
                            input_size=graph_representation.state_representation.size() * 2,
                          feature_size=graph_representation.num_nodes,
                          state_representation=graph_representation.state_representation,
                          param_groups=self.param_groups,
                          **kwargs
                          )

        # Backbone
        self.backbone = None

        # Dynamics
        self.dynamics = Linear(**kwargs)

    def forward(self, x: torch.Tensor, xb: torch.Tensor = None, y: torch.Tensor = None) -> Union[Tuple[torch.distributions.Distribution, dict], torch.distributions.Distribution]:
        if not self.training:
            y = None
        yb = None
        if self.backbone is not None:
            yb = self.backbone(xb)

        delta_q = qmul(x[:, 1:].contiguous(), inverse(x[:, :-1].contiguous()))
        delta_q = torch.cat([torch.zeros_like(delta_q[:, [0]]), delta_q], dim=1)
        x = torch.cat([x, delta_q], dim=-1)

        loc, log_Z, z, kwargs = self.core(x, yb, y)

        loc_q = loc.permute(0, 1, 3, 2, 4).contiguous()
        log_Z = log_Z.permute(0, 1, 3, 2, 4).contiguous()

        M, Z = self.to_bingham_M_Z(loc_q, log_Z)

        dist_q = BinghamMixtureModel(z, M, Z)

        q = GaussianMixtureModel.zero_variance(z, loc_q).mode
        p = self.graph_representation.to_position(q)

        dist_q_chained = ChainedBinghamMixtureModel(z, ChainedBingham(self._chain_list, Bingham(M, Z)))

        return dist_q_chained, {
            'dist_q': dist_q,
            'q': q,
            'p': p,
            'z': z,
            **kwargs
        }

    def hparams(self):
        return {}

    def to_bingham_M_Z(self, loc, log_Z):
        bs = list(loc.shape[:-1])
        M_uns = torch.stack([
            qmul(loc, torch.tensor([0, 1., 0, 0], device=loc.device).expand(bs + [4])),
            qmul(loc, torch.tensor([0, 0, 1., 0], device=loc.device).expand(bs + [4])),
            qmul(loc, torch.tensor([0, 0, 0, 1.], device=loc.device).expand(bs + [4])),
        ], dim=-2)
        log_z_desc, sort_idx = torch.sort(log_Z, dim=-1, descending=True)
        sort_idx = sort_idx.unsqueeze(-1).repeat([1] * (len(bs) + 1) + [4])
        M = M_uns.gather(dim=-2, index=sort_idx)
        M = torch.cat([M, qmul(loc, torch.tensor([1., 0, 0, 0], device=loc.device).expand(bs + [4])).unsqueeze(-2)],
                      dim=-2)
        Z = -torch.sigmoid(
            log_z_desc) * 900.  # force vanisihing gradient towards the limit so that the model can concentrate on the mean
        Z = torch.cat([Z, torch.zeros(Z.shape[:-1] + (1,), device=Z.device)], dim=-1)
        return M, Z
