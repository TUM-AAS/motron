import torch
import torch.nn as nn

import math

from motion.components.graph_linear import NodeLinear


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

class TransformerEncoder(nn.Module):
    def __init__(self,
                 graph_influence: torch.nn.Parameter,
                 node_types,
                 input_size: int,
                 output_size: int,
                 enc_num_layers: int = 1,
                 enc_activation_fn: object = None,
                 **kwargs):
        super().__init__()
        d_model = 21*output_size
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 3)
        self.pos_encoder = PositionalEncoding(d_model, 0.)
        self.fc = NodeLinear(graph_influence=graph_influence, in_features=input_size, out_features=output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, src):
        src = src.permute(1, 0, 2, 3)
        s_shape = src.shape

        src = self.fc(src) * math.sqrt(8)
        src = self.dropout(src)
        src = src.view(s_shape[0], s_shape[1], -1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.view(s_shape[0], s_shape[1], 21, -1).permute(1, 0, 2, 3)[:, -1]