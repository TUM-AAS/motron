import torch
from torch.distributions import Distribution, constraints, MixtureSameFamily

from motion.bingham import Bingham, Normalized


class JointVonMisesFisher(Distribution):
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
        for node_chaines in self._chaines:
            chained_log_prob_list.append(dist_log_prob[..., node_chaines, :].sum(dim=-2))

        return torch.stack(chained_log_prob_list, dim=-2)


class JointVonMisesFisherMixtureModel(MixtureSameFamily):
    pass
