import torch
import torch.nn as nn

from motion.core.mgs2s.decoder import Decoder
from motion.core.mgs2s.encoder import Encoder


class MGS2S(nn.Module):
    def __init__(self,
                 nodes: int,
                 graph_influence: torch.nn.Parameter,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 latent_size: int,
                 feature_size: int,
                 prediction_horizon: int,
                 param_groups=None,
                 **kwargs):
        super().__init__()
        self.param_groups = param_groups
        self._nodes = nodes
        self._latent_size = latent_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._feature_size = feature_size
        self._prediction_horizon = prediction_horizon
        self.encoder = Encoder(graph_influence=graph_influence, input_size=input_size, output_size=hidden_size, **kwargs)
        self.enc_to_z = nn.Linear(nodes*hidden_size, latent_size)
        self.decoder = Decoder(
                            graph_influence=graph_influence,
                                input_size=input_size,
                               hidden_size=hidden_size,
                               latent_size=latent_size,
                               output_size=output_size,
                               prediction_horizon=prediction_horizon,
                               param_groups=self.param_groups,
                               **kwargs)

    def forward(self, x: torch.Tensor, b: torch.tensor = None, y: torch.Tensor = None):
        bs = x.shape[0]
        enc = self.encoder(x)
        z = torch.distributions.Categorical(logits=self.enc_to_z(enc.flatten(start_dim=-2)).unsqueeze(1).unsqueeze(1).repeat(1, self._prediction_horizon, self._feature_size, 1))

        # Repeat encoded values for each latent mode
        z_all = torch.eye(self._latent_size).unsqueeze(0).unsqueeze(-2).repeat_interleave(repeats=bs, dim=0).repeat_interleave(repeats=enc.shape[-2], dim=-2).to(x.device)  # [bs, ls, ls]
        enc_tiled = enc.unsqueeze(1).repeat_interleave(repeats=self._latent_size, dim=1)  # [bs, ls, enc_s]
        x_tiled = x[:, -1].unsqueeze(1).repeat_interleave(repeats=self._latent_size, dim=1)

        # Permute to [bs, ts, ls, fs]
        loc, log_Z = self.decoder(x_tiled, enc_tiled, z_all, y)

        return loc, log_Z, z, {}
