from typing import Tuple, Union

import torch

from common.quaternion import inverse, qmul
from common.torch import Module
from motion.core.mgs2s.mgs2s import MGS2S
from motion.distributions import BinghamMixtureModel, GaussianMixtureModel
from motion.dynamics import Linear


class Motion(Module):
    def __init__(self, graph_representation, **kwargs):
        super(Motion, self).__init__()

        self.param_groups = [{
            'teacher_forcing_factor': 0.
        }]

        self.graph_representation = graph_representation

        self.graph_influence_matrix = torch.nn.Parameter(self.graph_representation.adjacency_matrix)
        # Core Model
        self.core = MGS2S(nodes=graph_representation.num_nodes,
                          graph_influence=self.graph_influence_matrix,
                            input_size=graph_representation.state_representation.size(),# * 2,
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

        #delta_q = qmul(x[:, 1:].contiguous(), inverse(x[:, :-1].contiguous()))
        #delta_q = torch.cat([torch.zeros_like(delta_q[:, [0]]), delta_q], dim=1)
        #x = torch.cat([x, delta_q], dim=-1)

        loc, log_Z, z, kwargs = self.core(x, yb, y)

        loc_q = loc.permute(0, 1, 3, 2, 4).contiguous()
        log_Z = log_Z.permute(0, 1, 3, 2, 4).contiguous()

        dist = BinghamMixtureModel.from_vector_params(z, loc_q, log_Z)

        loc_mode = GaussianMixtureModel.zero_variance(z, loc_q).mode
        loc_p = self.graph_representation.to_position(loc_mode)

        if self.training:
            return dist, {'z': z, 'quaternions': loc_mode, 'p': loc_p, **kwargs}
        return dist, {'quaternions': loc_mode, 'p': loc_p, **kwargs}

    def hparams(self):
        return {}
