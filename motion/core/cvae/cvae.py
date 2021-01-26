from typing import Union

import torch
import torch.nn as nn
from torch.distributions import MixtureSameFamily, Distribution, Categorical, kl_divergence

from motion.components.structural import StaticGraphLinear
from motion.core.mgs2s.decoder import Decoder
from motion.core.mgs2s.encoder import Encoder
from motion.utils.torch import mutual_inf_px


class CVAE(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 input_size: int,
                 encoder_hidden_size: int,
                 bottleneck_size: int,
                 decoder_hidden_size: int,
                 output_size: int,
                 latent_size: int,
                 prediction_horizon: int,
                 G: Union[torch.Tensor, torch.nn.Parameter] = None,
                 T: torch.Tensor = None,
                 param_groups=None,
                 **kwargs):
        super().__init__()
        self.param_groups = param_groups
        self._num_nodes = num_nodes
        self._latent_size = latent_size
        self._prediction_horizon = prediction_horizon

        self.encoder = Encoder(num_nodes=num_nodes,
                               input_size=input_size,
                               hidden_size=encoder_hidden_size,
                               output_size=bottleneck_size,
                               G=G,
                               T=T,
                               **kwargs)

        self.encoder_y = Encoder(num_nodes=num_nodes,
                                 input_size=input_size // 2,
                                 hidden_size=encoder_hidden_size,
                                 output_size=bottleneck_size,
                                 G=G,
                                 T=T,
                                 **kwargs)

        self.enc_to_z = nn.Linear(num_nodes * bottleneck_size, latent_size)

        self.p_z = StaticGraphLinear(bottleneck_size, bottleneck_size, num_nodes=num_nodes)
        self.q_z = StaticGraphLinear(2 * bottleneck_size, bottleneck_size, num_nodes=num_nodes)

        self.decoder = Decoder(num_nodes=num_nodes,
                               input_size=bottleneck_size,
                               feature_size=input_size,
                               hidden_size=decoder_hidden_size,
                               latent_size=latent_size,
                               output_size=output_size,
                               prediction_horizon=prediction_horizon,
                               G=G,
                               T=T,
                               param_groups=self.param_groups,
                               **kwargs)

    def forward(self, x: torch.Tensor, b: torch.tensor = None, y: torch.Tensor = None):
        bs = x.shape[0]

        # Encode History
        h, _, _ = self.encoder(x)  # [B, N, D]

        # Same z for all nodes and all timesteps
        if self.training:
            # Encode y
            h_y, _, _ = self.encoder_y(y)
            h = torch.cat([h, h_y], dim=-1)
            h = self.q_z(h)

            h_p = self.p_z(h)
            z_logits_p = self.enc_to_z(h_p.flatten(start_dim=-2)).unsqueeze(1).unsqueeze(1)  # [B, 1, 1 Z]
            z_logits_p = z_logits_p.repeat(1, self._prediction_horizon, self._num_nodes, 1)  # [B, T, N, Z]
            self.p_z_dist = torch.distributions.Categorical(logits=z_logits_p)

        else:
            h = self.p_z(h)
        z_logits = self.enc_to_z(h.flatten(start_dim=-2)).unsqueeze(1).unsqueeze(1)  # [B, 1, 1 Z]
        z_logits = z_logits.repeat(1, self._prediction_horizon, self._num_nodes, 1)  # [B, T, N, Z]
        p_z = torch.distributions.Categorical(logits=z_logits)

        # Sample all z
        z = torch.eye(self._latent_size, device=x.device)
        z = z.repeat(bs, 1).unsqueeze(-2).repeat_interleave(repeats=h.shape[-2], dim=-2)  # [B * Z, N, Z]

        # Repeat hidden for each |z|
        h = h.repeat_interleave(repeats=self._latent_size, dim=0)  # [B * Z, N, D]

        # Repeat last two D for each |z|
        x_tiled = x[:, -2:].repeat_interleave(repeats=self._latent_size, dim=0)  # [B * Z, N, D]

        # Repeat y for each |z|
        if y is not None:
            y = y.repeat_interleave(repeats=self._latent_size, dim=0)  # [B * Z, T, N, D]

        # Decode future q
        q, Z = self.decoder(x_tiled, h, z, y)  # [B * Z, T, N, D]

        # Reshape
        q = q.view((bs, -1) + q.shape[1:])  # [B, Z, T, N, D]
        Z = Z.view((bs, -1) + Z.shape[1:])  # [B, Z, T, N, D]

        return q, Z, p_z, {}

    def loss(self, y_pred: MixtureSameFamily, y: torch.Tensor) -> torch.Tensor:
        ll = y_pred.log_prob(y).sum(dim=1).mean()
        mi = mutual_inf_px(y_pred.mixture_distribution)
        if self.training:
            kl = kl_divergence(y_pred.mixture_distribution, self.p_z_dist).mean()
        else:
            kl = 0.
        return -ll - mi + self.param_groups[0]['kl_weight'] * kl
