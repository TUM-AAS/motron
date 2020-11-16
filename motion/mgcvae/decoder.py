import torch
import torch.nn as nn

from common.torch import get_activation_function


class Decoder(nn.Module):
    def __init__(self, feature_size: int,
                 latent_size: int,
                 enc_num_layers: int = 1,
                 enc_num_neurons: list = None,
                 enc_activation_fn: object = None):
        super(Decoder, self).__init__()
        torch.nn.GRU
        self.activation_fn = get_activation_function(enc_activation_fn)
        self.fcL = nn.ModuleList()
        if enc_num_layers > 1:
            assert enc_num_neurons is not None and len(enc_num_neurons) == enc_num_layers - 1
            input_size = feature_size
            for output_size in enc_num_neurons:
                self.fcL.append(nn.Linear(input_size, output_size))
                input_size = output_size
            self.fcL.append(nn.Linear(input_size, latent_size))
        else:
            self.fcL.append(nn.Linear(feature_size, latent_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc in self.fcL:
            x = self.activation_fn(fc(x))
        return x
