import torch

from motion.quaternion import Quaternion


class ToBingham(torch.nn.Module):
    def __init__(self, max_Z):
        super().__init__()
        self._max_Z = max_Z
        self.register_buffer('d1', torch.tensor([0, 1., 0, 0]))
        self.register_buffer('d2', torch.tensor([0, 0, 1., 0]))
        self.register_buffer('d3', torch.tensor([0, 0, 0, 1.]))

    def forward(self, q: torch.Tensor, Z: torch.tensor = None):
        bs = list(q.shape[:-1])
        M_uns = torch.stack([
            Quaternion.mul_(q, self.d1.expand(bs + [4])),
            Quaternion.mul_(q, self.d2.expand(bs + [4])),
            Quaternion.mul_(q, self.d3.expand(bs + [4])),
        ], dim=-2)

        # Sort Z descending
        Z_desc, sort_idx = torch.sort(Z, dim=-1, descending=True)
        sort_idx = sort_idx.unsqueeze(-1).repeat([1] * (len(bs) + 1) + [4])

        # Rearange M vectors according to Z order
        M = M_uns.gather(dim=-2, index=sort_idx)
        M = torch.cat([M, q.unsqueeze(-2)], dim=-2)

        # Force vanisihing gradient towards the limit of the lookup table
        Z = -torch.sigmoid(Z_desc) * (self._max_Z - 1e-3) - 1e-3
        Z = torch.cat([Z, torch.zeros(Z.shape[:-1] + (1,), device=Z.device)], dim=-1)
        return M, Z