from typing import Tuple

import torch
import torch.nn as nn

from common.torch import Module


class ToGMMParameter(Module):
    def __init__(self, input_size: int, output_state_size: int, dist_state_size: int, **kwargs):
        super().__init__()
        self.to_loc = nn.Linear(input_size // 2, output_state_size)
        self.to_log_dig = nn.Linear(input_size // 2, dist_state_size)
        self.to_tril = nn.Linear(input_size // 2, (dist_state_size * (dist_state_size - 1)) // 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1, x2 = x.split(x.shape[-1] // 2, dim=-1)
        return (self.to_loc(x1),
                self.to_log_dig(x2),
                self.to_tril(x2))
