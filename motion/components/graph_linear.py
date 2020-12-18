import math

import torch
from torch import nn


class GraphLinear(nn.Linear):
    def __init__(self, graph_influence, in_features: int, out_features: int, bias: bool = True) -> None:
        self.nodes = graph_influence.shape[-1]
        super().__init__(in_features, out_features, bias)
        self.G = graph_influence

        self.register_buffer('G_c', self.G)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        G = self.G_c
        assert x.shape[-2] == G.shape[-1]
        output = G.matmul(x.matmul(self.weight.t()))
        if self.bias is not None:
            output += self.bias
        return output


def ebmm(x, W):
    return torch.einsum('ndo,bnd->bno', (W, x))

def ebmmz(x, W):
    return torch.einsum('ndo,bznd->bzno', (W, x))


class NodeLinear(nn.Module):
    def __init__(self, graph_influence, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.G = torch.nn.Parameter(graph_influence)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(self.G.shape[0], out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #torch.nn.init.kaiming_uniform_(self.G, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 3:
            output = self.G.matmul(ebmm(input, self.weight.transpose(-2, -1)))
        else:
            output = self.G.matmul(ebmmz(input, self.weight.transpose(-2, -1)))
        if self.bias is not None:
            output += self.bias
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
