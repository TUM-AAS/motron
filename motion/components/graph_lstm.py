import torch
from torch import nn

import math


class GraphLSTM(nn.Module):
    def __init__(self, graph_influence, input_size, hidden_size):
        super().__init__()
        self.nodes = graph_influence.shape[-1]
        self.G = torch.nn.Parameter(graph_influence)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight is self.G:
                continue
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, init_states):
        """Assumes x is of shape (batch, sequence, nodes, feature)"""
        G = self.G
        assert x.shape[-2] == G.shape[-1]
        bs, seq_sz, _, _ = x.size()
        hidden_seq = []
        h_t, c_t = init_states


        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = G @ x_t @ self.W + G @ h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[..., :HS]),  # input
                torch.sigmoid(gates[..., HS:HS * 2]),  # forget
                torch.tanh(gates[..., HS * 2:HS * 3]),
                torch.sigmoid(gates[..., HS * 3:]),  # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)