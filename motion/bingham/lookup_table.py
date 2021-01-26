import os
import pathlib
import numpy as np
import torch


class LookupTable:
    class __LookupTable:
        def __init__(self):
            self.npz = np.load(os.path.join(pathlib.Path(__file__).parent.absolute(), 'bingham_lookups.npz'))
            self.tensor_dict = {}

        def __getitem__(self, item):
            if item not in self.tensor_dict:
                self.tensor_dict[item] = torch.Tensor(self.npz[item])
            return self.tensor_dict[item]

    instance = None

    def __init__(self):
        if not LookupTable.instance:
            LookupTable.instance = LookupTable.__LookupTable()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __getitem__(self, item):
        return self.instance.__getitem__(item)