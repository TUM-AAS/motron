from typing import Tuple, Union

import torch

from common.torch import Module
from motion.components.to_gmm_param import ToGMMParameter
from motion.core.mgs2s.mgs2s import MGS2S
from motion.distributions.gmm import GaussianMixtureModel
from motion.core.gru import GRU
from motion.dynamics import Linear


class Motion(Module):
    def __init__(self, node_representation, state_representation, **kwargs):
        super(Motion, self).__init__()

        self.node_representation = node_representation

        # Core Model
        self.core = MGS2S(input_size=node_representation.num_nodes() * state_representation.size(),
                          feature_size=node_representation.num_nodes(),
                          state_representation=state_representation,
                          **kwargs
                          )

        # Backbone
        self.backbone = None

        # Dynamics
        self.dynamics = Linear(**kwargs)

        self.to_gmm_params = ToGMMParameter(input_size=kwargs['output_size'] // node_representation.num_nodes(),
                                            output_state_size=state_representation.size(),
                                            **kwargs)

    def forward(self, x: torch.Tensor, xb: torch.Tensor = None, y: torch.Tensor = None) -> Union[Tuple[torch.distributions.Distribution, dict], torch.distributions.Distribution]:
        x_shape = x.shape
        bs = x_shape[0]
        ts = x_shape[1]
        fn = x_shape[2]
        fs = x_shape[3]
        feature_shape = list(x_shape[2:])
        feature_shape_len = len(feature_shape)

        x_f = x.flatten(start_dim=-feature_shape_len)

        if not self.training:
            y = None
        yb = None
        if self.backbone is not None:
            yb = self.backbone(xb)
        y, z, kwargs = self.core(x_f, yb, y)
        y = y.view(bs, y.shape[1], y.shape[-2], fn, -1).permute(0, 1, 3, 2, 4)
        loc, log_dig, tril = self.to_gmm_params(y)
        loc = x[:, [-1]].unsqueeze(-2) + torch.cumsum(loc, dim=1)
        dist = GaussianMixtureModel.from_vector_params(z, loc, log_dig, tril)

        if self.training:
            return dist, {'z': z, **kwargs}
        return dist

    def hparams(self):
        return {}
