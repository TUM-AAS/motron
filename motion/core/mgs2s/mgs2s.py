from typing import Union

import torch
import torch.nn as nn
from torch.distributions import MixtureSameFamily

from motion.components.structural import StaticGraphLinear
from motion.core.mgs2s.decoder import Decoder
from motion.core.mgs2s.encoder import Encoder
from motion.core.mgs2s.fc_decoder import FCDecoder
from motion.core.mgs2s.transformer_encoder import MyTransformerEncoder
from motion.utils.torch import mutual_inf_px


class MGS2S(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 input_size: int,
                 encoder_hidden_size: int,
                 bottleneck_size: int,
                 decoder_hidden_size: int,
                 output_size: int,
                 latent_size: int,
                 G: Union[torch.Tensor, torch.nn.Parameter] = None,
                 T: torch.Tensor = None,
                 param_groups=None,
                 **kwargs):
        super().__init__()
        self.param_groups = param_groups

        self._num_nodes = num_nodes
        self._latent_size = latent_size

        self.encoder = Encoder(num_nodes=num_nodes,
                               input_size=input_size,
                               hidden_size=encoder_hidden_size,
                               output_size=bottleneck_size,
                               G=G,
                               T=T,
                               **kwargs)

        self.enc_to_z = StaticGraphLinear(encoder_hidden_size,
                                          latent_size,
                                          bias=False,
                                          learn_influence=True,
                                          graph_influence = G,
                                          num_nodes=num_nodes,
                                          node_types=T)

        self.decoder = Decoder(num_nodes=num_nodes,
                               input_size=bottleneck_size,
                               feature_size=input_size,
                               hidden_size=decoder_hidden_size,
                               latent_size=latent_size,
                               output_size=output_size,
                               G=G,
                               T=T,
                               param_groups=self.param_groups,
                               **kwargs)

        self.z_dropout = nn.Dropout(0.)

    def forward(self, x: torch.Tensor, q: torch.Tensor, y: torch.Tensor = None, ph=1):
        bs = x.shape[0]

        # Encode History
        h, _, h_f = self.encoder(x)  # [B, N, D]

        # Same z for all nodes and all timesteps
        z_logits = self.z_dropout(self.enc_to_z(h_f)).unsqueeze(1)  # [B, 1, N Z]
        z_logits = z_logits.repeat(1, ph, 1, 1)  # [B, T, N, Z]
        p_z = torch.distributions.Categorical(logits=z_logits)

        # Sample all z
        z_mask = torch.eye(self._latent_size, device=x.device)
        z_mask = z_mask.repeat(bs, 1).unsqueeze(-2).repeat_interleave(repeats=h.shape[-2], dim=-2)  # [B * Z, N, Z]
        z = z_mask #- 0.5 #p_z.probs[:, 0].repeat_interleave(repeats=self._latent_size, dim=0) * z_mask

        # Repeat hidden for each |z|
        h = h.repeat_interleave(repeats=self._latent_size, dim=0)  # [B * Z, N, D]

        # Repeat last two D for each |z|
        x_tiled = x[:, -2:].repeat_interleave(repeats=self._latent_size, dim=0)  # [B * Z, N, D]

        q_t_tiled = q[:, -1].repeat_interleave(repeats=self._latent_size, dim=0)

        # Repeat y for each |z|
        if y is not None:
            y = y.repeat_interleave(repeats=self._latent_size, dim=0)  # [B * Z, T, N, D]

        # Decode future q
        q, Z, kwargs = self.decoder(x_tiled, h, z, q_t_tiled, y, ph)  # [B * Z, T, N, D]

        # Reshape
        q = q.view((bs, -1) + q.shape[1:])  # [B, Z, T, N, D]
        Z = Z.view((bs, -1) + Z.shape[1:])  # [B, Z, T, N, D]

        return q, Z, p_z, {**kwargs}

    def nll(self, y_pred: MixtureSameFamily, y: torch.Tensor) -> torch.Tensor:
        ll = y_pred.log_prob(y).sum(dim=1).mean()
        return -ll

    def loss(self, y_pred: MixtureSameFamily, y: torch.Tensor) -> torch.Tensor:
        nll = self.nll(y_pred, y)
        mi = mutual_inf_px(y_pred.mixture_distribution)
        return nll #- mi
