from typing import Tuple, Optional, List, Union

import torch
from torch.nn import *
import math

def gmm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.einsum('ndo,bnd->bno', w, x)


class GraphLinear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)
        #if self.learn_influence:
        #    self.G.data.uniform_(-stdv, stdv)
        if len(self.weight.shape) == 3:
            self.weight.data[1:] = self.weight.data[0]
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        if g is None and self.learn_influence:
            g = torch.nn.functional.normalize(self.G, p=1., dim=1)
            #g = torch.softmax(self.G, dim=1)
        elif g is None:
            g = self.G
        w = self.weight[self.node_type_index]
        output = self.mm(input, w.transpose(-2, -1))
        if self.bias is not None:
            bias = self.bias[self.node_type_index]
            output += bias
        output = g.matmul(output)

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
    def __init__(self, *args, bias: bool = True, num_nodes: int = None, graph_influence: Union[torch.Tensor, Parameter] = None,
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

        self.learn_influence = learn_influence

        if graph_influence is not None:
            assert num_nodes == graph_influence.shape[0] or num_nodes is None, 'Number of Nodes or Graph Influence Matrix has to be given.'
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

        if bias:
            if node_types is not None:
                self.bias = Parameter(torch.Tensor(num_node_types, self.out_features))
            else:
                self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

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
            assert num_nodes == graph_influence.shape[0] or num_nodes is None, 'Number of Nodes or Graph Influence Matrix has to be given.'
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
            self.register_buffer('node_type_index', node_types)
        else:
            self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
            self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
            self.mm = torch.matmul
            self.register_buffer('node_type_index', None)

        if bias:
            if node_types is not None:
                self.bias_ih = Parameter(torch.Tensor(num_node_types, 4 * hidden_size))
                self.bias_hh = Parameter(torch.Tensor(num_node_types, 4 * hidden_size))
            else:
                self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
                self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.bias_ih = None
            self.bias_hh = None

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
            gx = torch.nn.functional.normalize(self.G, p=1., dim=1)
            #gx = torch.softmax(self.G, dim=1)
        elif gx is None:
            gx = self.G

        hx = self.r_dropout(hx)

        weight_ih = self.weight_ih[self.node_type_index]
        weight_hh = self.weight_hh[self.node_type_index]
        if self.bias_hh is not None:
            bias_hh = self.bias_hh[self.node_type_index]
        else:
            bias_hh = 0.

        c_mask = (torch.remainder(torch.tensor(t + 1., device=input.device), self.phase) < 0.01).type_as(cx)

        gates = (self.dropout(self.mm(input, weight_ih.transpose(-2, -1))) +
                 self.mm(hx, weight_hh.transpose(-2, -1)) + bias_hh)
        gates = torch.matmul(gx, gates)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 2)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = c_mask * ((forgetgate * cx) + (ingate * cellgate)) + (1 - c_mask) * cx
        hy = outgate * torch.tanh(cy)

        gx = gx + self.G_add
        if self.learn_influence or self.learn_additive_graph_influence:
            gx = torch.nn.functional.normalize(gx, p=1., dim=1)
            #gx = torch.softmax(gx, dim=1)

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

GraphGRUState = Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]


class StaticGraphGRUCell_(Module):
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
            assert num_nodes == graph_influence.shape[0] or num_nodes is None, 'Number of Nodes or Graph Influence Matrix has to be given.'
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
            self.weight_ih = Parameter(torch.Tensor(num_node_types, 3 * hidden_size, input_size))
            self.weight_hh = Parameter(torch.Tensor(num_node_types, 3 * hidden_size, hidden_size))
            self.mm = gmm
            self.register_buffer('node_type_index', node_types)
        else:
            self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
            self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
            self.mm = torch.matmul
            self.register_buffer('node_type_index', None)

        if bias:
            if node_types is not None:
                self.bias_ih = Parameter(torch.Tensor(num_node_types, 3 * hidden_size))
                self.bias_hh = Parameter(torch.Tensor(num_node_types, 3 * hidden_size))
            else:
                self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
                self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.bias_ih = None
            self.bias_hh = None

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
            #if weight is self.weight_hh or weight is self.weight_ih and len(self.weight_ih.shape) == 3:
            #    weight.data[1:] = weight.data[0]

    def forward(self, input: torch.Tensor, state: GraphGRUState, t: int = 0) -> Tuple[torch.Tensor, GraphGRUState]:
        hx, gx = state
        if hx is None:
            hx = torch.zeros(input.shape[0], self.num_nodes, self.hidden_size, dtype=input.dtype, device=input.device)
        if gx is None and self.learn_influence:
            gx = torch.nn.functional.normalize(self.G, p=1., dim=1)
            #gx = torch.softmax(self.G, dim=1)
        elif gx is None:
            gx = self.G

        hx = self.r_dropout(hx)

        weight_ih = self.weight_ih[self.node_type_index]
        weight_hh = self.weight_hh[self.node_type_index]
        if self.bias_hh is not None:
            bias_hh = self.bias_hh[self.node_type_index]
        else:
            bias_hh = 0.
        if self.bias_ih is not None:
            bias_ih = self.bias_ih[self.node_type_index]
        else:
            bias_ih = 0.

        c_mask = (torch.remainder(torch.tensor(t + 1., device=input.device), self.phase) < 0.01).type_as(hx)

        x_results = self.dropout(self.mm(input, weight_ih.transpose(-2, -1))) + bias_ih
        h_results = self.mm(hx, weight_hh.transpose(-2, -1)) + bias_hh
        x_results = torch.matmul(gx, x_results)
        h_results = torch.matmul(gx, h_results)

        i_r, i_z, i_n = x_results.chunk(3, 2)
        h_r, h_z, h_n = h_results.chunk(3, 2)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)

        hy = n - torch.mul(n, z) + torch.mul(z, hx)
        hy = c_mask * hy + (1 - c_mask) * hx

        gx = gx + self.G_add
        if self.learn_influence or self.learn_additive_graph_influence:
            gx = torch.nn.functional.normalize(gx, p=1., dim=1)
            #gx = torch.softmax(gx, dim=1)

        return hy, (hy, gx)


class StaticGraphGRU_(Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, layer_dropout: float = 0.0, **kwargs):
        super().__init__()
        self.layers = ModuleList([StaticGraphGRUCell_(input_size, hidden_size, **kwargs)]
                                 + [StaticGraphGRUCell_(hidden_size, hidden_size, **kwargs) for _ in range(num_layers - 1)])
        self.dropout = Dropout(layer_dropout)

    def forward(self, input: torch.Tensor, states: Optional[List[GraphGRUState]] = None, t_i: int = 0) -> Tuple[torch.Tensor, List[GraphGRUState]]:
        if states is None:
            n: Optional[torch.Tensor] = None
            states = [(n, n)] * len(self.layers)

        output_states: List[GraphGRUState] = []
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


def StaticGraphGRU(*args, **kwargs):
    return torch.jit.script(StaticGraphGRU_(*args, **kwargs))
