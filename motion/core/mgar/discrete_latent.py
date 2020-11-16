import torch
import torch.nn as nn


class DiscreteLatent(nn.Module):
    def __init__(self, **kwargs):
        super(DiscreteLatent, self).__init__()


    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        q_z_c = self.encoder_q_z_c(c)
        if self.training:
            p_z_xc = self.encoder_p_z_xc(x, c)
            z = self.latent(p_z_xc)
        else:
            z = self.latent(q_z_c)
        p_y_zc = self.decoder(z)
        return p_y_zc
