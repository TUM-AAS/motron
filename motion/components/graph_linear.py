import torch
from torch import nn


class GraphLinear(nn.Linear):
    def __init__(self, graph_influence, in_features: int, out_features: int, bias: bool = True) -> None:
        self.nodes = graph_influence.shape[-1]
        super().__init__(in_features * self.nodes, out_features * self.nodes, bias)
        self.G = graph_influence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        G = self.G
        assert x.shape[-2] == G.shape[-1]
        x = x.flatten(start_dim=-2)
        GW = G.repeat_interleave(repeats=self.weight.shape[-2] // G.shape[-2], dim=-2)\
            .repeat_interleave(repeats=self.weight.shape[-1] // G.shape[-1], dim=-1)
        out = torch.nn.functional.linear(x, self.weight * GW, self.bias)
        return out.view(out.shape[:-1] + (self.nodes, -1))
