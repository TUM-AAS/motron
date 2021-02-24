from typing import Tuple, Union

import torch
import torch.nn as nn

from motion.components.node_dropout import NodeDropout
from motion.components.structural import StaticGraphLinear, StaticGraphLSTM, GraphLSTMState
from motion.utils.torch import get_activation_function


class Encoder(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 G: Union[torch.Tensor, torch.nn.Parameter] = None,
                 T: torch.Tensor = None,
                 enc_num_layers: int = 1,
                 dropout: float = 0.,
                 **kwargs):
        super().__init__()

        self.activation_fn = torch.nn.LeakyReLU()
        self.num_layers = enc_num_layers

        self.rnn = StaticGraphLSTM(input_size, hidden_size, num_layers=enc_num_layers, node_types=T, num_nodes=num_nodes, bias=True, clockwork=True)

        self.fc = StaticGraphLinear(hidden_size, output_size, num_nodes=num_nodes, node_types=T, bias=True)

        self.initial_hidden1 = StaticGraphLinear(input_size, hidden_size, num_nodes=num_nodes, node_types=T, bias=True)
        self.initial_hidden2 = StaticGraphLinear(input_size, hidden_size, num_nodes=num_nodes, node_types=T, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, state: torch.Tensor = None) -> Tuple[torch.Tensor, GraphLSTMState]:
        # Initialize hidden state of rnn
        if state is None:
            rnn_h1 = self.initial_hidden1(x[:, 0])
            rnn_h2 = self.initial_hidden2(x[:, 0])
            state = [(rnn_h1, rnn_h2, None)] * self.num_layers

        y, state = self.rnn(x, state)  # [B, T, N, D]
        h = self.activation_fn(self.fc(self.dropout(y[:, -1])))  # [B, N, D]
        return h, state, self.activation_fn(y[:, -1])
