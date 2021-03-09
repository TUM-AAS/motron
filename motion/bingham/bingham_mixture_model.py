import torch
from torch.distributions import MixtureSameFamily, Categorical

from motion.bingham import Bingham


class BinghamMixtureModel(MixtureSameFamily):
    def _pad_mixture_dimensions(self, x):
        dist_batch_ndims = len(self.batch_shape)
        cat_batch_ndims = len(self.mixture_distribution.batch_shape)
        pad_ndims = 0 if cat_batch_ndims == 1 else \
            dist_batch_ndims - cat_batch_ndims
        xs = x.shape
        x = x.reshape(xs[:-1] + torch.Size(pad_ndims * [1]) +
                      xs[-1:] + torch.Size(self._event_ndims * [1]))
        return x

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape

            # mixture samples [n, B]
            mix_sample = self.mixture_distribution.sample(sample_shape)
            mix_shape = mix_sample.shape

            # component samples [n, B, k, E]
            comp_samples = self.component_distribution.sample(sample_shape)
            comp_shape = comp_samples.shape

            # Gather along the k dimension
            mix_sample_r = mix_sample.reshape(
                mix_shape + torch.Size([1] * (len(es) + 1)))
            #mix_sample_r = mix_sample_r.repeat(
            #    torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es)

            mix_sample_r = mix_sample_r.repeat(
                torch.Size(torch.tensor(comp_shape[:len(mix_shape)]) // torch.tensor(mix_shape)) + torch.Size([1]) + es)

            samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
            return samples.squeeze(gather_dim)

    @property
    def mode(self):
        probs = self._pad_mixture_dimensions(
            torch.nn.functional.one_hot(
                self.mixture_distribution.probs.argmax(dim=-1), num_classes=self.mixture_distribution.probs.shape[-1]
            )
        )
        return torch.sum(probs * self.component_distribution.mean,
                         dim=-1 - self._event_ndims)  # [B, E]

    def to(self, device):
        return BinghamMixtureModel(mixture_distribution=Categorical(logits=self.mixture_distribution.logits.to(device)),
                                   component_distribution=Bingham(self.component_distribution.M.to(device), self.component_distribution.Z.to(device)))
