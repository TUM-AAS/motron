from typing import Tuple

import torch
import torch.nn as nn

from common.torch import Module


class ToBMMParameter(Module):
    def __init__(self, graph_influence, input_size: int, output_state_size: int, **kwargs):
        super().__init__()
        self.to_loc = nn.Linear(input_size, output_state_size - 1)
        self.to_log_Z = nn.Linear(input_size, output_state_size - 1)

    def forward(self, x1: torch.Tensor, x2) -> Tuple[torch.Tensor, torch.Tensor]:
        #x1, x2 = x.split(x.shape[-1] // 2, dim=-1)
        return self.to_loc(x1), self.to_log_Z(x2)
