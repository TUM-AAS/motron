from motion.state_representation import *


class Graph:
    def __init__(self, state_representation: str, initial_data=None):
        self._data = initial_data
        self.state_representation = eval(state_representation)

    @property
    def adjacency_matrix(self):
        return NotImplementedError

    @property
    def num_nodes(self):
        return NotImplementedError
