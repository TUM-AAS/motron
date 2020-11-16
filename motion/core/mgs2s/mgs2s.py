import torch
import torch.nn as nn

from motion.core.mgs2s.decoder import Decoder
from motion.core.mgs2s.encoder import Encoder


class MGS2S(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 latent_size: int,
                 feature_size: int,
                 prediction_horizon: int,
                 **kwargs):
        super().__init__()
        self._latent_size = latent_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._feature_size = feature_size
        self._prediction_horizon = prediction_horizon
        self.encoder = Encoder(input_size=input_size, output_size=hidden_size, **kwargs)
        self.enc_to_z = nn.Linear(hidden_size, latent_size)
        self.decoder = Decoder(input_size=input_size,
                               hidden_size=hidden_size,
                               latent_size=latent_size,
                               output_size=output_size,
                               prediction_horizon=prediction_horizon,
                               **kwargs)

    def forward(self, x: torch.Tensor, b: torch.tensor = None, y: torch.Tensor = None):
        bs = x.shape[0]
        enc = self.encoder(x)
        z = torch.distributions.Categorical(logits=self.enc_to_z(enc).unsqueeze(1).unsqueeze(1).repeat(1, self._prediction_horizon, self._feature_size, 1))

        # Repeat encoded values for each latent mode
        z_all = torch.eye(self._latent_size).unsqueeze(0).repeat(bs, 1, 1)  # [bs, ls, ls]
        enc_tiled = enc.unsqueeze(-2).repeat(1, self._latent_size, 1)  # [bs, ls, enc_s]
        x_tiled = x[:, [-1]].repeat(1, self._latent_size, 1)

        # Concat latent and encoded
        #z_enc = torch.cat([z_all, enc_rep], dim=-1).view(-1, 1, self._latent_size + self._hidden_size)

        # Permute to [bs, ts, ls, fs]
        y = self.decoder(x_tiled, enc_tiled, z_all)

        return y, z, {}
