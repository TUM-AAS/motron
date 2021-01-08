import torch
import torch.nn as nn

from common.torch import get_activation_function
from motion.components.graph_linear import GraphLinear, NodeLinear
from motion.components.graph_lstm import GraphLSTM, NodeLSTM, StackedNodeLSTM
from motion.components.structural import StaticGraphLinear, StaticGraphLSTM, BN


class Encoder(nn.Module):
    def __init__(self,
                 graph_influence: torch.nn.Parameter,
                 node_types,
                 input_size: int,
                 output_size: int,
                 enc_num_layers: int = 1,
                 enc_activation_fn: object = None,
                 **kwargs):
        super().__init__()
        self.activation_fn = get_activation_function(enc_activation_fn)
        self.rnn = StaticGraphLSTM(input_size, 256, num_layers=1, num_nodes=21, dropout=0., clockwork=False)
        self.fc = StaticGraphLinear(256, output_size, num_nodes=21)
        self.initial_hidden1 = StaticGraphLinear(input_size, 256, num_nodes=21)
        self.initial_hidden2 = StaticGraphLinear(input_size, 256, num_nodes=21)

        self.graph_influence = graph_influence

        self.dropout = nn.Dropout(0.)


    def forward(self, x: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
        bs = x.shape[0]
        if state is None:
            rnn_h1 = self.initial_hidden1(x[:, 0])
            rnn_h2 = self.initial_hidden2(x[:, 0])
            state = [(rnn_h1, rnn_h2, None)]
        # Initialize hidden state of rnn
        y, state = self.rnn(x, state)
        return self.dropout(torch.tanh(self.fc(y[:, -1]))), state, y[:, -1]
