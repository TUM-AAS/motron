import torch
from torch import nn


class GraphLinear(nn.Linear):
    def __init__(self, graph_influence, in_features: int, out_features: int, bias: bool = True) -> None:
        self.nodes = graph_influence.shape[-1]
        super().__init__(in_features, out_features, bias)
        self.G = torch.nn.Parameter(graph_influence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        G = self.G
        assert x.shape[-2] == G.shape[-1]
        output = G.matmul(x.matmul(self.weight.t()))
        if self.bias is not None:
            output += self.bias
        return output
