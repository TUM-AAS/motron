from typing import List

import numpy as np


class Skeleton:
    def __init__(self,
                 offsets: List,
                 parents: List,
                 joints_left: List,
                 joints_right: List):
        self.offsets = np.asarray(offsets)
        self.parents = np.asarray(parents)
        self.joints_left = np.asarray(joints_left)
        self.joints_right = np.asarray(joints_right)
        assert len(self.offsets.shape) == 2
        assert len(self.parents.shape) == 1
        assert self.offsets.shape[0] == self.parents.shape[0]

    def num_joints(self):
        return len(self.parents)
