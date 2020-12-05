import torch
import torch_bingham
from torch.distributions import Distribution, constraints


class Normalized(constraints.Constraint):
    """
    Trivially constrain to the extended real line `[-inf, inf]`.
    """
    def check(self, value):
        return (torch.abs(torch.norm(value, dim=-1) - 1) < 1e-6).all()


class Bingham(Distribution):
    arg_constraints = {'concentration': constraints.real}
    support = Normalized
    has_rsample = False

    def __init__(self, loc, lam):
        self.loc = loc
        self.lam = lam.clamp(1e-12, 900)
        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def log_prob(self, value):
        value_rep = value.repeat_interleave(repeats=self.loc.shape[-2], dim=-2)
        loc_view = self.loc.view(-1, self.loc.shape[-1])
        lam_view = self.lam.view(-1, self.lam.shape[-1])
        value_view = value_rep.view(-1, value_rep.shape[-1])
        log_prob_view = torch_bingham.bingham_prob(loc_view, -lam_view, value_view)
        log_prob = log_prob_view.view(self.loc.shape[:-1])
        return log_prob

    @property
    def mean(self):
        return self.loc



mu = torch.tensor([[1.,0, 0, 0]])
Zbatch = torch.tensor([[3., 2., 1.]])
Zbatch = torch.tensor([[1., 1., 900.]])

t = torch.tensor([[0.,0., 1., 0]])

dist = Bingham(mu, Zbatch)
p1 = dist.log_prob(mu)
p2 = dist.log_prob(t)
print(p1)
print(p2)