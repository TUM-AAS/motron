import torch
import torch.nn as nn

from motion.core.mgs2s.decoder import Decoder
from motion.core.mgs2s.encoder import Encoder
from motion.core.mgs2s.transformer_encoder import MyTransformerEncoder


class MGS2S(nn.Module):
    def __init__(self,
                 nodes: int,
                 graph_influence: torch.nn.Parameter,
                 node_types,
                 input_size: int,
                 encoder_hidden_size: int,
                 bottleneck_size: int,
                 decoder_hidden_size: int,
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
        self._output_size = output_size
        self._feature_size = feature_size
        self._prediction_horizon = prediction_horizon
        self.encoder = Encoder(graph_influence=graph_influence, node_types=node_types, input_size=input_size, hidden_size=encoder_hidden_size, output_size=bottleneck_size, **kwargs)
        self.enc_to_z = nn.Linear(nodes*bottleneck_size, latent_size)
        self.decoder = Decoder(
                            graph_influence=graph_influence,
                            node_types=node_types,
                                input_size=bottleneck_size,
                                feature_size=input_size,
                               hidden_size=decoder_hidden_size,
                               latent_size=latent_size,
                               output_size=output_size,
                               prediction_horizon=prediction_horizon,
                               param_groups=self.param_groups,
                               **kwargs)

    def forward(self, x: torch.Tensor, b: torch.tensor = None, y: torch.Tensor = None):
        bs = x.shape[0]
        enc, enc_s = self.encoder(x)
        z = torch.distributions.Categorical(
            logits=self.enc_to_z(enc.flatten(start_dim=-2)).unsqueeze(1).unsqueeze(1).repeat(1,
                                                                                             self._prediction_horizon,
                                                                                             self._feature_size, 1))
        # Repeat encoded values for each latent mode
        z_all = torch.eye(self._latent_size).repeat(bs, 1).unsqueeze(-2).repeat_interleave(
            repeats=enc.shape[-2], dim=-2).to(x.device)  # [bs, ls, ls]
        enc = enc.repeat_interleave(repeats=self._latent_size, dim=0)  # [bs, ls, enc_s]
        x_tiled = x[:, [-1]].repeat_interleave(repeats=self._latent_size, dim=0)
        # enc_s = [(h.repeat_interleave(repeats=self._latent_size, dim=0),
        #           c.repeat_interleave(repeats=self._latent_size, dim=0),
        #           g) for h, c, g in enc_s]
        loc_l = []
        loc_d_l = []
        log_Z_l = []
        # for pred in range(self._prediction_horizon // 5 - 1):
        #     loc, loc_d, log_Z = self.decoder(x_tiled, enc, z_all, y)
        #     enc, enc_s = self.encoder(torch.cat([loc, loc_d], dim=-1), enc_s)
        #     loc_l.append(loc)
        #     log_Z_l.append(log_Z)

        if y is not None:
            y = y.repeat_interleave(repeats=self._latent_size, dim=0)

        loc, loc_d, log_Z = self.decoder(x_tiled, enc, z_all, y)
        loc_l.append(loc)
        log_Z_l.append(log_Z)

        loc = torch.cat(loc_l, dim=1)
        log_Z = torch.cat(log_Z_l, dim=1)
        return loc.view((bs, -1) + loc.shape[1:]), log_Z.view((bs, -1) + log_Z.shape[1:]), z, {}
