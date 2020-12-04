from motion.state_representation import *


class GraphRepresentation:
    def __init__(self, state_representation: str, initial_data=None):
        self._data = initial_data
        self.state_representation = eval(state_representation)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, x):
        self._data = x
