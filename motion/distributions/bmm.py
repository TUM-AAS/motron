import torch
from torch.distributions import MultivariateNormal, MixtureSameFamily

from common.quaternion import qmul
from directional_distributions import Bingham


class BinghamMixtureModel(MixtureSameFamily):
    def __init__(self, mix, M, Z):
        self.mix = mix
        self.M = M
        self.Z = Z
        super().__init__(mix, Bingham(M=M, Z=Z))

    @classmethod
    def from_vector_params(cls, mix, loc, log_Z, **kwargs):
        bs = list(loc.shape[:-1])
        M_uns = torch.stack([
            qmul(loc, torch.tensor([0, 1., 0, 0], device=loc.device).expand(bs + [4])),
            qmul(loc, torch.tensor([0, 0, 1., 0], device=loc.device).expand(bs + [4])),
            qmul(loc, torch.tensor([0, 0, 0, 1.], device=loc.device).expand(bs + [4])),
        ], dim=-2)
        log_z_desc, sort_idx = torch.sort(log_Z, dim=-1, descending=True)
        sort_idx = sort_idx.unsqueeze(-1).repeat([1] * (len(bs) + 1) + [4])
        M = M_uns.gather(dim=-2, index=sort_idx)
        M = torch.cat([M, qmul(loc, torch.tensor([1., 0, 0, 0], device=loc.device).expand(bs + [4])).unsqueeze(-2)], dim=-2)
        Z = -torch.sigmoid(log_z_desc) * 900.  # force vanisihing gradient towards the limit so that the model can concentrate on the mean
        Z = torch.cat([Z, torch.zeros(Z.shape[:-1] + (1, ), device=Z.device)], dim=-1)
        return cls(mix, M, Z)

    def __getitem__(self, item):
        new_mix = torch.distributions.Categorical(logits=self.mixture_distribution.logits[item])
        return MixtureSameFamily(new_mix, Bingham(M=self.M[item], Z=self.Z[item]))

    @property
    def mode(self):
        probs = self._pad_mixture_dimensions(
            torch.nn.functional.one_hot(
                self.mixture_distribution.probs.argmax(dim=-1), num_classes=self.mixture_distribution.probs.shape[-1]
            )
        )
        return torch.sum(probs * self.component_distribution.mean,
                         dim=-1 - self._event_ndims)  # [B, E]