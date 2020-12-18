from typing import List, Tuple
import torch
from torch import nn
import torch.jit as jit

import math


def ebmm(x, W):
    return torch.einsum('ndo,bnd->bno', (W, x))




class GraphLSTMCell(jit.ScriptModule):
    def __init__(self, graph_influence, input_size, hidden_size, dropout=0., recurrent_dropout=0.):
        super().__init__()
        self.G = graph_influence
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = torch.nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.Tensor(4 * hidden_size))
        self.dropout = torch.nn.Dropout(dropout)
        self.r_dropout = torch.nn.Dropout(recurrent_dropout)
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            #if weight is self.G:
            #    continue
            weight.data.uniform_(-stdv, stdv)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        hx = self.r_dropout(hx)
        gates = (torch.matmul(self.G, self.dropout(torch.matmul(input, self.weight_ih.t()))) + self.bias_ih +
                 torch.matmul(self.G, torch.matmul(hx, self.weight_hh.t())) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 2)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

class NodeLSTMCell(jit.ScriptModule):
    def __init__(self, graph_influence, node_types, input_size, hidden_size, node_dropout=0., recurrent_dropout=0.):
        super().__init__()
        self.G = graph_influence
        self.add_G = torch.nn.Parameter(torch.zeros_like(graph_influence))
        self.t = node_types
        self.input_size = input_size
        self.hidden_size = hidden_size

        num_node_types = self.t.max() + 1
        self.weight_ih_org = torch.nn.Parameter(torch.Tensor(num_node_types, 4 * hidden_size, input_size))
        self.weight_hh_org = torch.nn.Parameter(torch.Tensor(num_node_types, 4 * hidden_size, hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.Tensor(4 * hidden_size))
        self.n_dropout = torch.nn.Dropout(node_dropout)
        self.r_dropout = torch.nn.Dropout(recurrent_dropout)
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight is self.G:
               continue
            if weight is self.add_G:
               continue
            # elif weight is self.weight_ih:
            #     weight.data = torch.zeros_like(weight[[0]]).uniform_(-stdv, stdv).repeat(weight.shape[0], 1, 1)
            # elif weight is self.weight_hh:
            #     weight.data = torch.zeros_like(weight[[0]]).uniform_(-stdv, stdv).repeat(weight.shape[0], 1, 1)
            weight.data.uniform_(-stdv, stdv)

    @jit.script_method
    def forward(self, input, state, G):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]
        hx, cx = state
        hx = self.r_dropout(hx)
        if len(G) == 0:
            G = self.G
        weight_ih = self.weight_ih_org[self.t]
        weight_hh = self.weight_hh_org[self.t]
        gates = (torch.matmul(G, self.n_dropout(ebmm(input, weight_ih.transpose(-2, -1)))) + self.bias_ih +
                 torch.matmul(G, ebmm(hx, weight_hh.transpose(-2, -1))) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 2)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        G = G + self.add_G

        return hy, (hy, cy), G


class GraphLSTM(jit.ScriptModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.cell = GraphLSTMCell(**kwargs)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(1)
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs, dim=1), state

class NodeLSTM(jit.ScriptModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.cell = NodeLSTMCell(**kwargs)

    @jit.script_method
    def forward(self, input, state, G):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]
        inputs = input.unbind(1)
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        for i in range(len(inputs)):
            out, state, G = self.cell(inputs[i], state, G)
            outputs += [out]
        return torch.stack(outputs, dim=1), state, G


class StackedNodeLSTM(jit.ScriptModule):
    #__constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, layer_args, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([NodeLSTM(**kwargs) for kwargs in layer_args])
        self.dropout = torch.nn.Dropout(dropout)

    @jit.script_method
    def forward(self, input, states, Gs):
        # type: (Tensor, List[Tuple[Tensor, Tensor]], List[Tensor]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]], List[Tensor]]
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[torch.Tensor, torch.Tensor]], [])
        output_Gs = jit.annotate(List[torch.Tensor], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            G = Gs[i]
            output, out_state, G = rnn_layer(output, state, G)
            output = self.dropout(output)
            output_states += [out_state]
            output_Gs += [G]
            i += 1
        return output, output_states, output_Gs

class StackedGraphLSTM(jit.ScriptModule):
    #__constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, layer_args, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([GraphLSTM(**kwargs) for kwargs in layer_args])
        self.dropout = torch.nn.Dropout(dropout)

    @jit.script_method
    def forward(self, input, states):
        # type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[torch.Tensor, torch.Tensor]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output = self.dropout(output)
            output_states += [out_state]
            i += 1
        return output, output_states


class GraphLSTMo(nn.Module):
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