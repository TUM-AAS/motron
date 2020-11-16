import torch
import torch.nn as nn

from common.torch import get_activation_function

# TODO: Idea: Can you use the inverse of the toGMM to populate the first input to the GRU e.g. control->state

class Decoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 latent_size: int,
                 output_size: int,
                 prediction_horizon: int,
                 dec_num_layers: int = 1,
                 dec_activation_fn: object = None,
                 **kwargs):
        super().__init__()
        self._prediction_horizon = prediction_horizon
        self.activation_fn = get_activation_function(dec_activation_fn)
        self.rnn = nn.LSTM(input_size=latent_size + output_size + hidden_size, hidden_size=output_size, num_layers=dec_num_layers, batch_first=True)
        self.fc = nn.Linear(output_size, output_size)
        self.initial_input = nn.Linear(input_size, output_size)
        self.initial_hidden1 = nn.Linear(latent_size + output_size + hidden_size, output_size)
        self.initial_hidden2 = nn.Linear(latent_size + output_size + hidden_size, output_size)
        self.h0 = nn.Parameter(torch.zeros(dec_num_layers, 1, output_size).normal_(std=0.01), requires_grad=True)
        self.dropout = nn.Dropout(0.)

    def forward(self, x: torch.Tensor, enc: torch.Tensor,  z: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]

        out = []
        u0 = self.initial_input(x)
        # Initialize hidden state of rnn
        xi = torch.cat([z, u0, enc], dim=-1).view(-1, 1, z.shape[-1] + u0.shape[-1] + enc.shape[-1])
        rnn_h1 = self.initial_hidden1(xi).view(1, xi.shape[0], -1)#self.h0.expand(-1, xi.shape[0], -1).contiguous()
        rnn_h2 = self.initial_hidden2(xi).view(1, xi.shape[0], -1)
        for i in range(self._prediction_horizon):
            rnn_out, (rnn_h1, rnn_h2) = self.rnn(xi, (rnn_h1, rnn_h2))
            yi = (rnn_out).view(bs, -1, rnn_out.shape[-1])
            #yi = self.activation_fn(self.fc(self.dropout(yi)))
            out.append(yi)
            xi = torch.cat([z, yi, enc], dim=-1).view(-1, 1, z.shape[-1] + yi.shape[-1] + enc.shape[-1])
        return torch.stack(out, dim=1)
