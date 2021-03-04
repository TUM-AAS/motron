import torch
from torch.distributions import Distribution, constraints, MixtureSameFamily

from motion.bingham import Bingham, Normalized


class JointBingham(Distribution):
    arg_constraints = {'concentration': constraints.real}
    support = Normalized
    has_rsample = False

    def __init__(self, chains: list, dist: Bingham):
        self._chaines = chains
        self._dist = dist
        super().__init__(batch_shape=dist.batch_shape, event_shape=dist.event_shape)

    def log_prob(self, value):
        dist_log_prob = self._dist.log_prob(value)

        chained_log_prob_list = []
        assert len(self._chaines) == dist_log_prob.shape[-2]
        for i, node_chaines in enumerate(self._chaines):
            chained_log_prob_list.append(1.*dist_log_prob[..., i, :])#.sum(dim=-2))

        return torch.stack(chained_log_prob_list, dim=-2)


class JointBinghamMixtureModel(MixtureSameFamily):
    def log_prob(self, x):
        x = self._pad(x)
        log_prob_x = 3.5*self.component_distribution.log_prob(x)  # [S, B, k]
        log_mix_prob = torch.log_softmax(self.mixture_distribution.logits,
                                         dim=-1)  # [B, k]
        return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]
