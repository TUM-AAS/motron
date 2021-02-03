import torch


class NodeDropout(torch.nn.Module):
    def __init__(self, p):
        self.p = p
        super().__init__()

    def forward(self, x):
        if not self.training:
            return x

        bs, ts, ns, ds = x.shape

        ind_mask = (torch.rand((bs, 1, ns, 1), device=x.device) < self.p).type_as(x)
        td_mask = torch.zeros((1, ts, 1, 1), device=x.device)
        tdi = torch.randint(ts, (1,), device=x.device)
        td_mask[:, -tdi:] = 1.
        mask_final = 1 - (ind_mask * td_mask)

        return x * mask_final