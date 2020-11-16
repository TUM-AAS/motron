import torch
import torch.nn as nn

from common.torch import get_activation_function


class Decoder(nn.Module):
    def __init__(self, latent_size: int,
                 output_size: int,
                 dec_num_layers: int = 1,
                 dec_activation_fn: object = None):
        super(Decoder, self).__init__()
        self.activation_fn = get_activation_function(dec_activation_fn)
        self.rnn = nn.GRU(input_size=latent_size, hidden_size=output_size, num_layers=dec_num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_fn(self.rnn(x))
        return x