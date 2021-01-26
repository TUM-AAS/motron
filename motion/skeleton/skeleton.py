from typing import List

import numpy as np
import torch


class Skeleton(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def num_joints(self):
        raise NotImplementedError()

    def forward(self, rotations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()












