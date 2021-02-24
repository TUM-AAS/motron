from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import *

import math

from torch.nn.modules.transformer import _get_activation_fn, _get_clones

from motion.components.structural import StaticGraphLSTM, StaticGraphLinear


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
        #src = query + self.dropout1(src2)
        return src2


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


class MyTransformerEncoder(Module):
    def __init__(self,
                 num_nodes: int,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 G: Union[torch.Tensor, torch.nn.Parameter] = None,
                 T: torch.Tensor = None,
                 enc_num_layers: int = 1,
                 dropout: float = 0.,
                 activation_fn: object = None,
                 **kwargs):
        super().__init__()

        self.activation_fn = torch.nn.LeakyReLU()
        self.num_layers = enc_num_layers

        self.rnn = StaticGraphLSTM(input_size, hidden_size//2, num_layers=self.num_layers, num_nodes=num_nodes, dropout=0., node_types=T, bias=False)

        self.fc = StaticGraphLinear(hidden_size, output_size, num_nodes=num_nodes, node_types=T, bias=False)

        self.initial_hidden1 = StaticGraphLinear(input_size, hidden_size//2, num_nodes=num_nodes, node_types=T, bias=False)
        self.initial_hidden2 = StaticGraphLinear(input_size, hidden_size//2, num_nodes=num_nodes, node_types=T, bias=False)

        self.dropout = Dropout(0.0)

        self.pe = PositionalEncoding(4, dropout=0.)

        #self.ln = torch.nn.LayerNorm(output_size, elementwise_affine=False)

        encoder_layers = TransformerLayer(d_model=21*hidden_size//2, nhead=1, dropout=0.0, kdim=21*hidden_size//2+4)
        self.transformer_encoder = TransformerEncoderML(encoder_layers, 1)

    def forward(self, x: torch.Tensor, state: torch.Tensor = None):
        bs, ts, ns, fs = x.shape

        vn = ts - 20 + 1
        vl = 20

        idx = torch.arange(vl).unsqueeze(0) + torch.arange(vn).unsqueeze(1)

        parts = x[:, idx].view(-1, vl, ns, fs)

        # Initialize hidden state of rnn
        if state is None:
            rnn_h1 = self.initial_hidden1(parts[:, 0])
            rnn_h2 = self.initial_hidden2(parts[:, 0])
            state = [(rnn_h1, rnn_h2, None)] * self.num_layers

        y, state = self.rnn(parts, state)  # [B, T, N, D]
        y = y.view(bs, vn, vl, ns, -1)[:, :, -1]
        #y = self.fc(y.view(-1, y.shape[-2], y.shape[-1]))
        value = y.permute(1, 0, 2, 3).flatten(start_dim=-2)
        query = value[[-1]]
        key = self.pe(value)

        h_t = self.transformer_encoder(query, key, value)[0].view(bs, ns, -1)
        h_t = torch.cat([h_t, y[:, -1]], dim=-1)
        h = self.activation_fn(self.fc(h_t))  # [B, N, D]
        return h, state, self.activation_fn(h_t)