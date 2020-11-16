from typing import Type

import torch
import torch.nn as nn

from common.torch import get_activation_function, Module
from motion.state_representation import StateRepresentation


class GRU(Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int,
                 activation_fn: object,
                 state_representation: Type[StateRepresentation],
                 prediction_horizon: int = 1,
                 dropout: float = 0.0,
                 **kwargs):
        super().__init__()
        self._state_representation = state_representation
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._activation_fn = activation_fn
        self._prediction_horizon = prediction_horizon

        self.prediction_horizon = prediction_horizon
        self.activation_fn = get_activation_function(activation_fn)
        self.rnn = nn.GRU(input_size=self._input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, self._output_size)

        self.h0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size).normal_(std=0.01), requires_grad=True)

        self.dropout = nn.Dropout(dropout)

        self.param_groups = [{
            'teacher_forcing_factor': 0.
        }]

    def forward(self, x: torch.Tensor, b: torch.tensor = None, y: torch.Tensor = None):
        x_shape = x.shape
        feature_shape = x_shape[2:]
        feature_shape_len = len(feature_shape)
        bs = x_shape[0]

        # Prepare output list
        out = []
        out_unnorm = []

        # Initialize hidden state of rnn
        rnn_h = self.h0.expand(-1, bs, -1).contiguous()

        # Get input state
        s = x

        # Set last state
        last_s = x[:, -1:].contiguous()

        for i in range(self.prediction_horizon):
            # Iterate rnn
            rnn_out, rnn_h = self.rnn(s.flatten(start_dim=-feature_shape_len), rnn_h)
            dx = self.activation_fn(self.fc(self.dropout(rnn_out[:, -1:])))

            # Reshape dx to feature shape
            dx = dx.view([bs, -1] + list(feature_shape))

            s = self._state_representation.sum(last_s, dx)
            if self.training and y is not None:
                teacher_forcing_mask = (torch.rand(list(s.shape[:-feature_shape_len]) + [1] * feature_shape_len)
                                        < self.param_groups[0]['teacher_forcing_factor'])
                last_s = teacher_forcing_mask.type_as(y) * y[:, [i]] + (~teacher_forcing_mask).type_as(y) * s
            else:
                last_s = s
            out.append(s)  # Stack along time dimension
        y = torch.cat(out, dim=1)
        return y.contiguous(), {}

    def hparams(self) -> dict:
        return {
            'GRU_input_size': self._input_size,
            'GRU_hidden_size': self._hidden_size,
            'GRU_num_layers': self._num_layers,
            'GRU_output_size': self._output_size,
            'GRU_activation_fn': self._activation_fn,
            'GRU_dropout': self._dropout,
            'GRU_prediction_horizon': self._prediction_horizon,
        }
