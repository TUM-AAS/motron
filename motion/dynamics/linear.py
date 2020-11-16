import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, **kwargs):
        super(Linear, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x