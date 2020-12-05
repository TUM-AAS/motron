import torch
import torch.nn as nn

from common.torch import get_activation_function
from motion.components.graph_linear import GraphLinear
from motion.components.graph_lstm import GraphLSTM


class Encoder(nn.Module):
    def __init__(self,
                 graph_influence: torch.nn.Parameter,
                 input_size: int,
                 output_size: int,
                 enc_num_layers: int = 1,
                 enc_activation_fn: object = None,
                 **kwargs):
        super().__init__()
        self.activation_fn = get_activation_function(enc_activation_fn)
        self.rnn = GraphLSTM(graph_influence=graph_influence, input_size=input_size, hidden_size=output_size)
        self.fc = GraphLinear(graph_influence=graph_influence, in_features=output_size, out_features=output_size)
        self.initial_hidden1 = GraphLinear(graph_influence=graph_influence, in_features=input_size, out_features=output_size)
        self.initial_hidden2 = GraphLinear(graph_influence=graph_influence, in_features=input_size, out_features=output_size)

        self.graph_influence = graph_influence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]

        rnn_h1 = self.initial_hidden1(x[:, 0])
        rnn_h2 = self.initial_hidden2(x[:, 0])
        hidden = (rnn_h1, rnn_h2)
        # Initialize hidden state of rnn
        y, _ = self.rnn(x, hidden)
        return torch.nn.functional.leaky_relu(self.fc(y[:, -1]))
