from typing import Tuple

import torch
import torch.nn as nn

from common.torch import Module
from motion.components.graph_linear import NodeLinear
from motion.components.structural import StaticGraphLinear


class ToBMMParameter(Module):
    def __init__(self, graph_influence, node_types, input_size: int, output_state_size: int, **kwargs):
        super().__init__()
        self.to_loc = StaticGraphLinear(input_size, output_state_size - 1, num_nodes=21, node_types=node_types)#NodeLinear(graph_influence, input_size, output_state_size - 1)
        self.to_log_Z = StaticGraphLinear(input_size, output_state_size - 1, num_nodes=21, node_types=node_types)#NodeLinear(graph_influence, input_size, output_state_size - 1)

    def forward(self, x1: torch.Tensor, x2) -> Tuple[torch.Tensor, torch.Tensor]:
        #x1, x2 = x.split(x.shape[-1] // 2, dim=-1)
        return self.to_loc(x1), self.to_log_Z(x2)
