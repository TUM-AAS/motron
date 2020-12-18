import torch
import torch.nn as nn

import math

from motion.components.graph_linear import NodeLinear
from motion.components.to_bingham_param import ToBMMParameter


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerDecoder(nn.Module):
    def __init__(self,
                 graph_influence: torch.nn.Parameter,
                 node_types,
                 input_size: int,
                 hidden_size: int,
                 latent_size: int,
                 output_size: int,
                 prediction_horizon: int,
                 state_representation,
                 dec_num_layers: int = 1,
                 dec_activation_fn: object = None,
                 param_groups=None,
                 **kwargs):
        super().__init__()
        self.param_groups = param_groups
        self._state_representation = state_representation
        self._prediction_horizon = prediction_horizon
        d_model = 21*output_size
        encoder_layers = nn.TransformerDecoderLayer(d_model=d_model, nhead=4)
        self.transformer_encoder = nn.TransformerDecoder(encoder_layers, 3)
        self.pos_encoder = PositionalEncoding(d_model, 0.)
        self.fc = nn.Linear(in_features=9, out_features=output_size)

        self.fc1 = nn.Linear(in_features=output_size, out_features=output_size)

        self.to_bmm_params = ToBMMParameter(graph_influence,
                                            output_size,
                                            output_state_size=4,
                                            dist_state_size=4,
                                            **kwargs)

    def forward(self, x: torch.Tensor, enc: torch.Tensor, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        nodes = x.shape[2]
        out = []
        out_log_diag = []

        loc_start = x[..., :4].clone()
        xi = torch.cat([z, x], dim=-1).view(-1, 1, x.shape[-2], z.shape[-1] + x.shape[-1])

        for i in range(self._prediction_horizon):
            yi = self.fc(xi) * math.sqrt(8)
            yi = yi.permute(1, 0, 2, 3)
            yi = yi.flatten(start_dim=-2)
            yi = self.pos_encoder(yi)
            yi = self.transformer_encoder(yi, enc[-i-1:].squeeze(1))[[-1]].permute(1, 0, 2)
            yi = yi.view(bs, z.shape[-1], nodes, -1)
            yi1 = torch.tanh(self.fc1(torch.tanh(yi)))
            yi2 = yi1.detach()
            loc, log_Z = self.to_bmm_params(yi1, yi2)
            w = torch.ones(loc.shape[:-1] + (1,), device=loc.device)
            loc = torch.cat([w, loc], dim=-1)
            loc_d = self._state_representation.validate(loc)
            loc = self._state_representation.sum(loc_start, loc_d)
            out.append(loc)
            out_log_diag.append(log_Z)
            if self.training and y is not None:
                teacher_forcing_mask = (torch.rand(list(loc.shape[:-2]) + [1] * 2)
                                        < self.param_groups[0]['teacher_forcing_factor'])
                loc_start = teacher_forcing_mask.type_as(y) * y[:, [i]] + (~teacher_forcing_mask).type_as(y) * loc
            else:
                loc_start = loc
            xin = torch.cat([z, loc_start, loc_d], dim=-1).view(-1, 1, x.shape[-2], z.shape[-1] + x.shape[-1])
            xi = torch.cat([xi, xin], dim=1)
        return torch.stack(out, dim=1).contiguous(), torch.stack(out_log_diag, dim=1).contiguous()