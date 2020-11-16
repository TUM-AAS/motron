import torch
import torch.nn as nn

from common.torch import get_activation_function


class Encoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 enc_num_layers: int = 1,
                 enc_activation_fn: object = None,
                 **kwargs):
        super().__init__()
        self.activation_fn = get_activation_function(enc_activation_fn)
        self.rnn = nn.GRU(input_size=input_size, hidden_size=output_size, num_layers=enc_num_layers, batch_first=True)
        self.h0 = nn.Parameter(torch.zeros(enc_num_layers, 1, output_size).normal_(std=0.01), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        # Initialize hidden state of rnn
        rnn_h = self.h0.expand(-1, bs, -1).contiguous()
        y, _ = self.rnn(x, rnn_h)
        return self.activation_fn(y[:, -1])
