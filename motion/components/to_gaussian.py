import torch

from motion.quaternion import Quaternion


class ToGaussian(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_diag, scale_tril_v):
        diag = log_diag #+ 1e-3#torch.exp(log_diag) # gradient of exp becomes nan for very small values
        #diag = torch.ones_like(diag) * torch.linspace(0.1, 3.16, 25, device=diag.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        scale_tril = torch.diag_embed(diag)
        tril_indices = torch.tril_indices(row=diag.shape[-1], col=diag.shape[-1], offset=-1)
        scale_tril[..., tril_indices[0], tril_indices[1]] = torch.zeros_like(scale_tril_v)

        return scale_tril