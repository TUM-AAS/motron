import torch
from torch.distributions import MixtureSameFamily

from motion.quaternion import Quaternion


class TimeSeriesMixtureModel(MixtureSameFamily):
    def __init__(self, *args, **kwargs):
        super().__init__(validate_args=False, *args, **kwargs)

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

    @property
    def weighted_mean(self):
        weights = self.mixture_distribution.probs.unsqueeze(-1)
        means = self.component_distribution.mean
        Q = (weights*means)
        QQT = Q.transpose(-1, -2) @ Q
        # There is a bug calculating the eigenvectors on GPU in torch 1.8
        mean_unnorm = torch.symeig(QQT.to('cpu'), eigenvectors=True)[1][..., -1].to(means.device)
        mean = torch.nn.functional.normalize(mean_unnorm, dim=-1)
        return mean

    def mode_opt(self, it=10, step=1e-3):
        with torch.enable_grad():
            means = self.component_distribution.mean.clone().permute(3, 0, 1, 2, 4)
            d_q = torch.randn(means.shape[:-1] + (3,), device=means.device) * 1e-4
            x = torch.nn.Parameter(d_q)
            opt = torch.optim.Adam([x], lr=step)
            for i in range(it):
                opt.zero_grad()
                dq_t = Quaternion(angle=torch.norm(x, dim=-1), axis=x).q
                q_t = Quaternion.mul_(dq_t, means)
                nll = -self.log_prob(q_t).sum(dim=2).mean(dim=-1).mean(dim=0).mean(dim=0)
                nll.backward()
                opt.step()

        dq_t = Quaternion(angle=torch.norm(x, dim=-1), axis=x).q
        new_means = Quaternion.mul_(dq_t, means)
        _, max_idx = torch.max(self.log_prob(new_means), dim=0)
        mode = new_means.gather(index=max_idx.unsqueeze(0).unsqueeze(-1).repeat_interleave(4, -1), dim=0).squeeze(0)
        return mode

    @property
    def mode_max(self):
        lp = self.log_prob(self.component_distribution.mean.permute(3, 0, 1, 2, 4))
        _, max_idx = lp.max(dim=0)
        max_idx = max_idx.unsqueeze(-1).unsqueeze(-1).repeat_interleave(4, dim=-1)
        mode = self.component_distribution.mean.gather(dim=-2, index=max_idx)
        return mode.squeeze(-2)

    def closest_mode(self, y):
        lp = self.component_distribution.log_prob(y.unsqueeze(-2))
        _, max_idx = lp.max(dim=-1)
        max_idx = max_idx.unsqueeze(-1).unsqueeze(-1).repeat_interleave(4, dim=-1)
        mode = self.component_distribution.mean.gather(dim=-2, index=max_idx)
        return mode.squeeze(-2)
