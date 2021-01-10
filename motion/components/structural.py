from typing import Tuple, Optional, List, Union

import torch
from torch.nn import *
import math

def gmm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.einsum('ndo,bnd->bno', w, x)


class GraphLinear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if len(self.weight.shape) == 3:
            self.weight.data[1:] = self.weight.data[0]
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, g: torch.Tensor = None) -> torch.Tensor:
        if g is None:
            g = self.G
        w = self.weight[self.node_type_index]
        output = g.matmul(self.mm(input, w.transpose(-2, -1)))# / g.shape[-1]
        if self.bias is not None:
            output += self.bias
        return output


class DynamicGraphLinear(GraphLinear):
    def __init__(self, num_node_types: int = 1, *args):
        super().__init__(*args)

    def forward(self, input: torch.Tensor, g: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
        assert g is not None or t is not None, "Either Graph Influence Matrix or Node Type Vector is needed"
        if g is None:
            g = self.G[t][:, t]
        return super().forward(input, g)



class StaticGraphLinear(GraphLinear):
    def __init__(self, *args, num_nodes: int = None, graph_influence: Union[torch.Tensor, Parameter] = None,
                 learn_influence: bool = False, node_types: torch.Tensor = None, weights_per_type: bool = False):
        """
        :param in_features: Size of each input sample
        :param out_features: Size of each output sample
        :param num_nodes: Number of nodes.
        :param graph_influence: Graph Influence Matrix
        :param learn_influence: If set to ``False``, the layer will not learn an the Graph Influence Matrix.
        :param node_types: List of Type for each node. All nodes of same type will share weights.
                Default: All nodes have unique types.
        :param weights_per_type: If set to ``False``, the layer will not learn weights for each node type.
        :param bias: If set to ``False``, the layer will not learn an additive bias.
        """
        super().__init__(*args)

        if graph_influence is not None:
            assert num_nodes is None, 'Number of Nodes or Graph Influence Matrix has to be given.'
            num_nodes = graph_influence.shape[0]
            if type(graph_influence) is Parameter:
                assert learn_influence, "Graph Influence Matrix is a Parameter, therefore it must be learnable."
                self.G = graph_influence
            elif learn_influence:
                self.G = Parameter(graph_influence)
            else:
                self.register_buffer('G', graph_influence)
        else:
            assert num_nodes, 'Number of Nodes or Graph Influence Matrix has to be given.'
            eye_influence = torch.eye(num_nodes, num_nodes)
            if learn_influence:
                self.G = Parameter(eye_influence)
            else:
                self.register_buffer('G', eye_influence)

        if weights_per_type and node_types is None:
            node_types = torch.tensor([i for i in range(num_nodes)])
        if node_types is not None:
            num_node_types = node_types.max() + 1
            self.weight = Parameter(torch.Tensor(num_node_types, self.out_features, self.in_features))
            self.mm = gmm
            self.node_type_index = node_types
        else:
            self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
            self.mm = torch.matmul
            self.node_type_index = None

        self.reset_parameters()



GraphLSTMState = Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]

class BN(Module):
    def __init__(self, num_nodes, num_features):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.bn = BatchNorm1d(num_nodes * num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x.view(-1, self.num_nodes * self.num_features)).view(-1, self.num_nodes, self.num_features)

class LinearX(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input

class StaticGraphLSTMCell_(Module):
    def __init__(self, input_size: int, hidden_size: int, num_nodes: int = None, dropout: float = 0.,
                 recurrent_dropout: float = 0., graph_influence: Union[torch.Tensor, Parameter] = None,
                 learn_influence: bool = False, additive_graph_influence: Union[torch.Tensor, Parameter] = None,
                 learn_additive_graph_influence: bool = False, node_types: torch.Tensor = None,
                 weights_per_type: bool = False, clockwork: bool = False, bias: bool = True):
        """

        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param num_nodes:
        :param dropout:
        :param recurrent_dropout:
        :param graph_influence:
        :param learn_influence:
        :param additive_graph_influence:
        :param learn_additive_graph_influence:
        :param node_types:
        :param weights_per_type:
        :param bias:
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.learn_influence = learn_influence
        self.learn_additive_graph_influence = learn_additive_graph_influence
        if graph_influence is not None:
            assert num_nodes is None, 'Number of Nodes or Graph Influence Matrix has to be given.'
            num_nodes = graph_influence.shape[0]
            if type(graph_influence) is Parameter:
                assert learn_influence, "Graph Influence Matrix is a Parameter, therefore it must be learnable."
                self.G = graph_influence
            elif learn_influence:
                self.G = Parameter(graph_influence)
            else:
                self.register_buffer('G', graph_influence)
        else:
            assert num_nodes, 'Number of Nodes or Graph Influence Matrix has to be given.'
            eye_influence = torch.eye(num_nodes, num_nodes)
            if learn_influence:
                self.G = Parameter(eye_influence)
            else:
                self.register_buffer('G', eye_influence)

        if additive_graph_influence is not None:
            if type(additive_graph_influence) is Parameter:
                self.G_add = additive_graph_influence
            elif learn_additive_graph_influence:
                self.G_add = Parameter(additive_graph_influence)
            else:
                self.register_buffer('G_add', additive_graph_influence)
        else:
            if learn_additive_graph_influence:
                self.G_add = Parameter(torch.zeros_like(self.G))
            else:
                self.G_add = 0.

        if weights_per_type and node_types is None:
            node_types = torch.tensor([i for i in range(num_nodes)])
        if node_types is not None:
            num_node_types = node_types.max() + 1
            self.weight_ih = Parameter(torch.Tensor(num_node_types, 4 * hidden_size, input_size))
            self.weight_hh = Parameter(torch.Tensor(num_node_types, 4 * hidden_size, hidden_size))
            self.mm = gmm
            self.node_type_index = node_types
        else:
            self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
            self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
            self.mm = torch.matmul
            self.node_type_index = None

        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.bias_ih = 0.
            self.bias_hh = 0.

        self.clockwork = clockwork
        if clockwork:
            phase = torch.arange(0., hidden_size)
            phase = phase - phase.min()
            phase = (phase / phase.max()) * 8.
            phase += 1.
            phase = torch.floor(phase)
            self.register_buffer('phase', phase)
        else:
            phase = torch.ones(hidden_size)
            self.register_buffer('phase', phase)

        self.dropout = Dropout(dropout)
        self.r_dropout = Dropout(recurrent_dropout)

        self.num_nodes = num_nodes

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight is self.G:
                continue
            if weight is self.G_add:
                continue
            weight.data.uniform_(-stdv, stdv)
            if weight is self.weight_hh or weight is self.weight_ih and len(self.weight_ih.shape) == 3:
                weight.data[1:] = weight.data[0]

    def forward(self, input: torch.Tensor, state: GraphLSTMState, t: int = 0) -> Tuple[torch.Tensor, GraphLSTMState]:
        hx, cx, gx = state
        if hx is None:
            hx = torch.zeros(input.shape[0], self.num_nodes, self.hidden_size, dtype=input.dtype, device=input.device)
        if cx is None:
            cx = torch.zeros(input.shape[0], self.num_nodes, self.hidden_size, dtype=input.dtype, device=input.device)
        if gx is None and self.learn_influence:
            gx = self.G / self.G.sum(dim=0, keepdim=True)
        else:
            gx = self.G

        hx = self.r_dropout(hx)

        weight_ih = self.weight_ih[self.node_type_index]
        weight_hh = self.weight_hh[self.node_type_index]

        c_mask = (torch.remainder(torch.tensor(t + 1., device=input.device), self.phase) < 0.01).type_as(cx)

        gates = (torch.matmul(gx, self.dropout(self.mm(input, weight_ih.transpose(-2, -1)))) + self.bias_ih +
                 torch.matmul(gx, self.mm(hx, weight_hh.transpose(-2, -1))) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 2)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = c_mask * ((forgetgate * cx) + (ingate * cellgate)) + (1 - c_mask) * cx
        hy = outgate * torch.tanh(cy)

        gx = gx + self.G_add
        if self.learn_influence or self.learn_additive_graph_influence:
            gx = gx / gx.sum(dim=0, keepdim=True)

        return hy, (hy, cy, gx)


class StaticGraphLSTM_(Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, layer_dropout: float = 0.0, **kwargs):
        super().__init__()
        self.layers = ModuleList([StaticGraphLSTMCell_(input_size, hidden_size, **kwargs)]
                                 + [StaticGraphLSTMCell_(hidden_size, hidden_size, **kwargs) for _ in range(num_layers - 1)])
        self.dropout = Dropout(layer_dropout)

    def forward(self, input: torch.Tensor, states: Optional[List[GraphLSTMState]] = None, t_i: int = 0) -> Tuple[torch.Tensor, List[GraphLSTMState]]:
        if states is None:
            n: Optional[torch.Tensor] = None
            states = [(n, n, n)] * len(self.layers)

        output_states: List[GraphLSTMState] = []
        output = input
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            inputs = output.unbind(1)
            outputs: List[torch.Tensor] = []
            for t, input in enumerate(inputs):
                out, state = rnn_layer(input, state, t_i+t)
                outputs += [out]
            output = torch.stack(outputs, dim=1)
            output = self.dropout(output)
            output_states += [state]
            i += 1
        return output, output_states


def StaticGraphLSTM(*args, **kwargs):
    return torch.jit.script(StaticGraphLSTM_(*args, **kwargs))


GraphRNNState = Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
class StaticGraphRNNCell_(Module):
    def __init__(self, input_size: int, hidden_size: int, num_nodes: int = None, dropout: float = 0.,
                 recurrent_dropout: float = 0., graph_influence: Union[torch.Tensor, Parameter] = None,
                 learn_influence: bool = False, additive_graph_influence: Union[torch.Tensor, Parameter] = None,
                 learn_additive_graph_influence: bool = False, node_types: torch.Tensor = None,
                 weights_per_type: bool = False, clockwork_bins: int = 1, bias: bool = True):

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if graph_influence is not None:
            assert num_nodes is None, 'Number of Nodes or Graph Influence Matrix has to be given.'
            num_nodes = graph_influence.shape[0]
            if type(graph_influence) is Parameter:
                assert learn_influence, "Graph Influence Matrix is a Parameter, therefore it must be learnable."
                self.G = graph_influence
            elif learn_influence:
                self.G = Parameter(graph_influence)
            else:
                self.register_buffer('G', graph_influence)
        else:
            assert num_nodes, 'Number of Nodes or Graph Influence Matrix has to be given.'
            eye_influence = torch.eye(num_nodes, num_nodes)
            if learn_influence:
                self.G = Parameter(eye_influence)
            else:
                self.register_buffer('G', eye_influence)

        if additive_graph_influence is not None:
            if type(additive_graph_influence) is Parameter:
                self.G_add = additive_graph_influence
            elif learn_additive_graph_influence:
                self.G_add = Parameter(additive_graph_influence)
            else:
                self.register_buffer('G_add', additive_graph_influence)
        else:
            if learn_additive_graph_influence:
                self.G_add = Parameter(torch.zeros_like(self.G))
            else:
                self.G_add = 0.

        if weights_per_type and node_types is None:
            node_types = torch.tensor([i for i in range(num_nodes)])
        if node_types is not None:
            num_node_types = node_types.max() + 1
            self.weight_ih = Parameter(torch.Tensor(num_node_types, hidden_size, input_size))
            self.weight_hh = Parameter(torch.Tensor(num_node_types, hidden_size, hidden_size))
            self.weight_ho = Parameter(torch.Tensor(num_node_types, hidden_size, hidden_size))
            self.mm = gmm
            self.node_type_index = node_types
        else:
            self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
            self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
            self.weight_ho = Parameter(torch.Tensor(hidden_size, hidden_size))
            self.mm = torch.matmul
            self.node_type_index = None

        if bias:
            self.bias_ih = Parameter(torch.Tensor(hidden_size))
            self.bias_hh = Parameter(torch.Tensor(hidden_size))
            self.bias_ho = Parameter(torch.Tensor(hidden_size))
        else:
            self.bias_ih = 0.
            self.bias_hh = 0.
            self.bias_ho = 0.

        phase = torch.square(torch.arange(0., hidden_size))
        phase = phase - phase.min()
        phase = (phase / phase.max()) * 10.
        phase += 1.
        phase = torch.floor(phase)
        self.register_buffer('phase', phase)

        self.dropout = Dropout(dropout)
        self.r_dropout = Dropout(recurrent_dropout)

        self.num_nodes = num_nodes

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight is self.G:
                continue
            if weight is self.G_add:
                continue
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, state: GraphRNNState, t: int = 0) -> Tuple[torch.Tensor, GraphRNNState]:
        hx, gx = state
        if hx is None:
            hx = torch.zeros(input.shape[0], self.num_nodes, self.hidden_size, dtype=input.dtype, device=input.device)
        if gx is None:
            gx = self.G

        hx = self.r_dropout(hx)

        weight_ih = self.weight_ih[self.node_type_index]
        weight_hh = self.weight_hh[self.node_type_index]
        weight_ho = self.weight_ho[self.node_type_index]

        c_mask = torch.remainder(torch.tensor(t + 1., device=input.device), self.phase) < 0.01

        h = (torch.matmul(gx, self.dropout(self.mm(input, weight_ih.transpose(-2, -1)))) + self.bias_ih +
                 torch.matmul(gx, self.mm(hx, weight_hh.transpose(-2, -1))) + self.bias_hh)

        hy = c_mask * torch.tanh(h) + (1. - c_mask * hx)
        o = torch.tanh(self.mm(h, weight_ho.transpose(-2, -1)) + self.bias_ho)

        gx = gx + self.G_add

        return o, (hy, gx)


class StaticGraphRNN_(Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, layer_dropout: float = 0.0, **kwargs):
        super().__init__()
        self.layers = ModuleList([StaticGraphRNNCell_(input_size, hidden_size, **kwargs)]
                                 + [StaticGraphRNNCell_(hidden_size, hidden_size, **kwargs) for _ in range(num_layers - 1)])
        self.dropout = Dropout(layer_dropout)

    def forward(self, input: torch.Tensor, states: Optional[List[GraphRNNState]] = None, t_i: int = 0) -> Tuple[torch.Tensor, List[GraphRNNState]]:
        if states is None:
            n: Optional[torch.Tensor] = None
            states = [(n, n)] * len(self.layers)

        output_states: List[GraphRNNState] = []
        output = input
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            inputs = output.unbind(1)
            outputs: List[torch.Tensor] = []
            for t, input in enumerate(inputs):
                out, state = rnn_layer(input, state, t_i+t)
                outputs += [out]
            output = torch.stack(outputs, dim=1)
            output = self.dropout(output)
            output_states += [state]
            i += 1
        return output, output_states


def StaticGraphRNN(*args, **kwargs):
    return torch.jit.script(StaticGraphRNN_(*args, **kwargs))


def test_graph_lstm():
    import os
    os.environ['PYTORCH_JIT'] = '0'
    m = StaticGraphLSTM(2, 20, 2, 0.1, num_nodes=3)
    l = StaticGraphLinear(2, 20, num_nodes=3)
    i = torch.rand((8, 5, 3, 2))
    li = torch.rand((8, 3, 2))
    o = l(li)
    o = m(i)

    print(o)