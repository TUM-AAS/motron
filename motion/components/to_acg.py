import torch

from motion.quaternion import Quaternion


class ToACG(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, r: torch.Tensor, Z: torch.tensor = None):
        bs = q.shape[:-1]
        M = torch.eye(4, device=q.device).expand(bs + (4, 4))
        M = Quaternion.mul_(q.unsqueeze(-2), M)
        M = Quaternion(r.unsqueeze(-2))._rotate_quaternion(Quaternion(M.transpose(-2, -1))).q.transpose(-2, -1)
        Z = torch.sigmoid(Z)
        Z = torch.cat([torch.ones_like(Z[..., [0]]), Z], dim=-1)

        return M, Z