import torch
from torch.distributions import MixtureSameFamily


class VonMisesFisherMixtureModel(MixtureSameFamily):
    @property
    def mode(self):
        probs = self._pad_mixture_dimensions(
            torch.nn.functional.one_hot(
                self.mixture_distribution.probs.argmax(dim=-1), num_classes=self.mixture_distribution.probs.shape[-1]
            )
        )
        return torch.sum(probs * self.component_distribution.mean,
                         dim=-1 - self._event_ndims)  # [B, E]