import torch
import torch.nn as nn

from common.torch import get_activation_function


class Encoder(nn.Module):
    def __init__(self, feature_size: int,
                 latent_size: int,
                 enc_num_layers: int = 1,
                 enc_activation_fn: object = None):
        super(Encoder, self).__init__()
        self.activation_fn = get_activation_function(enc_activation_fn)
        self.rnn = nn.GRU(input_size=feature_size, hidden_size=latent_size, num_layers=enc_num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_fn(self.rnn(x))
        return x
