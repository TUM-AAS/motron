import torch
from torch import nn

import math


class GraphLSTM(nn.Module):
    def __init__(self, graph_influence, input_size, hidden_size):
        super().__init__()
        self.nodes = graph_influence.shape[-1]
        self.G = graph_influence
        input_size = self.nodes * input_size
        hidden_size = self.nodes * hidden_size
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
        GW_ = G.repeat_interleave(repeats=self.W.shape[-2] // (G.shape[-2]), dim=-2)\
            .repeat_interleave(repeats=self.W.shape[-1] // (4 * G.shape[-1]), dim=-1)
        GU_ = G.repeat_interleave(repeats=self.U.shape[-2] // (G.shape[-2]), dim=-2) \
            .repeat_interleave(repeats=self.U.shape[-1] // (4 * G.shape[-1]), dim=-1)
        GW = GW_.repeat([1] * (len(GW_.shape) - 1) + [4])
        GU = GU_.repeat([1] * (len(GU_.shape) - 1) + [4])
        x = x.flatten(start_dim=-2)
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        h_t, c_t = init_states
        h_t = h_t.flatten(start_dim=-2)
        c_t = c_t.flatten(start_dim=-2)


        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ (self.W * GW) + h_t @ (self.U * GU) + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq.view(hidden_seq.shape[:-1] + (self.nodes, -1)),\
               (h_t.view(h_t.shape[:-1] + (self.nodes, -1)), c_t.view(c_t.shape[:-1] + (self.nodes, -1)))