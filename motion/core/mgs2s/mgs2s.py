from typing import Union

import torch
import torch.nn as nn
from torch.distributions import MixtureSameFamily

from motion.components.structural import StaticGraphLinear
from motion.core.mgs2s.decoder import Decoder
from motion.core.mgs2s.encoder import Encoder


class MGS2S(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 input_size: int,
                 encoder_hidden_size: int,
                 bottleneck_size: int,
                 decoder_hidden_size: int,
                 output_size: int,
                 latent_size: int,
                 position: bool = False,
                 G: Union[torch.Tensor, torch.nn.Parameter] = None,
                 T: torch.Tensor = None,
                 param_groups=None,
                 **kwargs):
        super().__init__()
        self.param_groups = param_groups

        self._num_nodes = num_nodes
        self._latent_size = latent_size
        self._position = position

        self.encoder = Encoder(num_nodes=num_nodes,
                               input_size=input_size,
                               hidden_size=encoder_hidden_size,
                               output_size=bottleneck_size,
                               G=G,
                               T=T,
                               **kwargs)

        self.enc_to_z = StaticGraphLinear(bottleneck_size,
                                          latent_size,
                                          bias=False,
                                          learn_influence=True,
                                          graph_influence=G,
                                          num_nodes=num_nodes,
                                          node_types=T)

        self.decoder = Decoder(num_nodes=num_nodes,
                               input_size=bottleneck_size,
                               feature_size=input_size,
                               hidden_size=decoder_hidden_size,
                               latent_size=latent_size,
                               output_size=output_size,
                               position=position,
                               G=G,
                               T=T,
                               param_groups=self.param_groups,
                               **kwargs)

        self.z_dropout = nn.Dropout(kwargs['dropout'])

    def forward(self, q: torch.Tensor, q_dot: torch.Tensor, p: torch.Tensor, p_dot: torch.Tensor, ph=1, state=None):
        bs = q.shape[0]

        if state is None:
            state = (None, None)

        # Encode History
        x = torch.cat([q, q_dot], dim=-1)
        if self._position:
            p_dot_padded = torch.nn.functional.pad(p_dot, (0, 5)).unsqueeze(2)
            x = torch.cat([x, p_dot_padded], dim=-2)
        h, encoder_state = self.encoder(x, state[0])  # [B, N, D]

        # Same z for all nodes and all timesteps
        z_logits = self.z_dropout(self.enc_to_z(h).mean(dim=-2))

        # Sample all z
        z_mask = torch.eye(self._latent_size, device=q.device)
        #z_mask = (z_mask.repeat(bs, 1) * z_logits.repeat(self._latent_size, 1)).unsqueeze(-2).repeat_interleave(repeats=h.shape[-2], dim=-2)
        z_mask = z_mask.repeat(bs, 1).unsqueeze(-2).repeat_interleave(repeats=h.shape[-2], dim=-2)  # [B * Z, N, Z]
        z = z_mask

        # Repeat hidden for each |z|
        h = h.repeat_interleave(repeats=self._latent_size, dim=0)  # [B * Z, N, D]

        # Repeat last two D for each |z|
        x_tiled = x[:, -2:].repeat_interleave(repeats=self._latent_size, dim=0)  # [B * Z, N, D]

        q_t_tiled = q[:, -1].repeat_interleave(repeats=self._latent_size, dim=0)
        if self._position:
            p_t_tiled = p[:, -1].repeat_interleave(repeats=self._latent_size, dim=0)
        else:
            p_t_tiled = None

        # Decode future q
        dq, dq_cov_lat, dp, dp_cov_lat, decoder_state = self.decoder(x=x_tiled,
                                                                     h=h,
                                                                     z=z,
                                                                     q_t=q_t_tiled,
                                                                     p_t=p_t_tiled,
                                                                     ph=ph,
                                                                     state=state[1])  # [B * Z, T, N, D]

        # Reshape
        dq = dq.view((bs, -1) + dq.shape[1:])  # [B, Z, T, N, D]
        dq_cov_lat = dq_cov_lat.view((bs, -1) + dq_cov_lat.shape[1:])  # [B, Z, T, N, D]
        if self._position:
            dp = dp.view((bs, -1) + dp.shape[1:])  # [B, Z, T, N, D]
            dp_cov_lat = dp_cov_lat.view((bs, -1) + dp_cov_lat.shape[1:])  # [B, Z, T, N, D]

        return dq, dq_cov_lat, dp, dp_cov_lat, z_logits, (encoder_state, decoder_state), {}

    def loss(self, y_pred: MixtureSameFamily, y: torch.Tensor) -> torch.Tensor:
        ll = ((y_pred.log_prob(y)).sum(dim=1)  # T
              .mean(-1)  # N
              .mean())
        return -ll
