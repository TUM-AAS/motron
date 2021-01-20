import math
from typing import Optional

import torch
import torch.nn as nn

from common.torch import get_activation_function

# TODO: Idea: Can you use the inverse of the toGMM to populate the first input to the GRU e.g. control->state
from directional_distributions import Bingham
from motion.components.graph_linear import NodeLinear
from motion.components.graph_lstm import GraphLSTM, NodeLSTM, StackedNodeLSTM
from motion.components.structural import GraphLinear, StaticGraphLinear, StaticGraphLSTM, StaticGraphRNN
from motion.components.to_bingham_param import ToBMMParameter
from motion.components.to_gmm_param import ToGMMParameter


class MyTransformerDecoder(nn.Module):
    def __init__(self,
                 graph_influence: torch.nn.Parameter,
                 node_types,
                 feature_size: int,
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
        self.activation_fn = get_activation_function(dec_activation_fn)
        self.rnn = StaticGraphLSTM(feature_size + 8, hidden_size, num_layers=1, graph_influence=graph_influence, learn_influence=True, node_types=node_types, dropout=0., recurrent_dropout=0., clockwork=False, learn_additive_graph_influence=True)

        self.fc = StaticGraphLinear(hidden_size, output_size, graph_influence=graph_influence, learn_influence=False, node_types=node_types)
        self.fc2 = StaticGraphLinear(hidden_size, output_size, graph_influence=graph_influence, learn_influence=False, node_types=node_types)
        self.initial_hidden1 = StaticGraphLinear(latent_size + input_size, hidden_size, graph_influence=graph_influence, node_types=node_types)
        self.initial_hidden2 = StaticGraphLinear(latent_size + input_size, hidden_size, graph_influence=graph_influence, node_types=node_types)
        self.dropout = nn.Dropout(0.)
        self.dropout1 = nn.Dropout(0.)

        self.fc_hd = Linear(hidden_size, 8)

        self.to_bmm_params = ToBMMParameter(graph_influence,
                                            node_types,
                                            output_size,
                                            output_state_size=4,
                                            dist_state_size=4,
                                            **kwargs)

        encoder_layers = TransformerLayer(d_model=8, nhead=2, dropout=0.1, kdim=8)
        self.transformer_encoder = TransformerEncoderML(encoder_layers, 1)

        self.af = torch.nn.LeakyReLU()

        self.pe = PositionalEncoding(d_model=4)

        self.ln1 = nn.LayerNorm(output_size)
        self.ln = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor, enc: torch.Tensor, full_enc,  z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bs, ts, ns, fs = x.shape
        out = []
        loc_d_l = []
        out_log_diag = []
        #u0 = self.initial_input(x)
        # Initialize hidden state of rnn
        loc_start = x[..., :4].clone()

        xi = x
        enc_z = torch.cat([z, enc], dim=-1)
        rnn_h1 = self.initial_hidden1(enc_z)
        rnn_h2 = self.initial_hidden2(enc_z)
        hidden = [(rnn_h1, rnn_h2, None)]

        t_values = full_enc.permute(1, 0, 2, 3).reshape(25, -1, 8)
        t_keys = t_values

        for i in range(self._prediction_horizon):
            att_xi = self.transformer_encoder(self.fc_hd(hidden[0][1]).unsqueeze(0).reshape(1, -1, 8), t_keys, t_values).view(bs, 1, ns, 8)
            xi = torch.cat([xi, att_xi], dim=-1)
            rnn_out, hidden = self.rnn((xi), hidden, i)
            yi = (rnn_out).squeeze(1)
            yi1 = torch.relu(self.fc(self.dropout(torch.relu(yi))))
            yi2 = torch.relu(self.fc2(self.dropout1(torch.relu(yi))))
            loc, log_Z = self.to_bmm_params(yi1, yi2)
            log_Z = log_Z * 100
            w = torch.ones(loc.shape[:-1] + (1,), device=loc.device)
            loc = loc
            loc = torch.cat([w, loc], dim=-1)
            loc_d = self._state_representation.validate(loc)
            loc = self._state_representation.sum(loc_start, loc_d)
            out.append(loc)
            loc_d_l.append(loc_d)
            out_log_diag.append(log_Z)
            if self.param_groups[0]['teacher_forcing_factor'] > 1e-6 and self.training and y is not None:
                teacher_forcing_mask = (torch.rand(list(loc.shape[:-2]) + [1] * 2)
                                        < self.param_groups[0]['teacher_forcing_factor'])
                loc_start = teacher_forcing_mask.type_as(y) * y[:, i] + (~teacher_forcing_mask).type_as(y) * loc
            else:
                loc_start = loc
            xi = torch.cat([loc_start, loc_d], dim=-1).unsqueeze(1)
        return torch.stack(out, dim=1).contiguous(), torch.stack(loc_d_l, dim=1).contiguous(), torch.stack(out_log_diag, dim=1).contiguous()



import torch
from torch import Tensor
from torch.nn import *

from torch.nn.modules.transformer import _get_activation_fn, _get_clones

class PositionalEncoding(Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = torch.cat([self.pe[:x.size(0), :].repeat(1, x.shape[1], 1), x], dim=-1)
        return x

class TransformerEncoderML(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoderML, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, key, value, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = query

        for mod in self.layers:
            output = mod(query, key, value, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", **kwargs):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, **kwargs)
        self.dropout1 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.functional.relu
        super(TransformerLayer, self).__setstate__(state)

    def forward(self, query, key, value, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(query, key, value, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = query + self.dropout1(src2)
        return src
