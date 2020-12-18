from typing import Tuple, Union

import torch
import numpy as np

from common.quaternion import inverse, qmul, qdist
from common.torch import Module
from directional_distributions import Bingham
from motion.core.mgs2s.mgs2s import MGS2S
from motion.distributions import BinghamMixtureModel, GaussianMixtureModel
from motion.distributions.chained_bingham import ChainedBinghamMixtureModel, ChainedBingham
from motion.dynamics import Linear


class Motion(Module):
    def __init__(self, graph_representation, **kwargs):
        super(Motion, self).__init__()

        self.param_groups = [{
            'teacher_forcing_factor': 0.
        }]

        self.graph_representation = graph_representation

        self.graph_influence_matrix = self.graph_representation.adjacency_matrix[self.graph_representation.dynamic_nodes][: ,self.graph_representation.dynamic_nodes]
        self.node_types = self.graph_representation.nodes_type_id_dynamic
        self.node_types = torch.tensor([i for i in range(len(self.node_types))])
        self._chain_list = self.graph_representation.chain_list

        # Core Model
        self.core = MGS2S(nodes=graph_representation.num_dynamic_nodes,
                          graph_influence=self.graph_influence_matrix,
                          node_types=self.node_types,
                            input_size=graph_representation.state_representation.size() * 2,
                          feature_size=graph_representation.num_dynamic_nodes,
                          state_representation=graph_representation.state_representation,
                          param_groups=self.param_groups,
                          **kwargs
                          )

        # Backbone
        self.backbone = None

        # Dynamics
        self.dynamics = Linear(**kwargs)

    def forward(self, x: torch.Tensor, xb: torch.Tensor = None, y: torch.Tensor = None) -> Union[Tuple[torch.distributions.Distribution, dict], torch.distributions.Distribution]:
        x = x[..., self.graph_representation.dynamic_nodes, :]

        if y is not None:
            y = y[..., self.graph_representation.dynamic_nodes, :]
        if not self.training:
            y = None
        yb = None
        if self.backbone is not None:
            yb = self.backbone(xb)

        delta_q = qmul(x[:, 1:].contiguous(), inverse(x[:, :-1].contiguous()))
        delta_q = torch.cat([torch.zeros_like(delta_q[:, [0]]), delta_q], dim=1)
        x = torch.cat([x, delta_q], dim=-1)

        # if self.training:
        #     mask = (0.5 > torch.rand_like(x[:, [0]][..., [0]]))
        #     zero_rot = torch.zeros_like(x)
        #     zero_rot[..., 0] = 1.
        #     zero_rot[..., 4] = 1.
        #     x = mask * x + (~mask) * zero_rot

        # if self.training:
        #     x_rand = torch.randn_like(x) * 0.03
        #     x_rand[..., 0] = 1.
        #     x_rand[..., 4] = 1.
        #     x_rand[..., :4] = torch.nn.functional.normalize(x_rand[..., :4], dim=-1)
        #     x_rand[..., 4:] = torch.nn.functional.normalize(x_rand[..., 4:], dim=-1)
        #     x[..., :4] = qmul(x[..., :4], x_rand[..., :4])
        #     x[..., 4:] = qmul(x[..., 4:], x_rand[..., 4:])

        loc, log_Z, z, kwargs = self.core(x, yb, y)

        z_logits = z.logits

        loc_q = loc.permute(0, 1, 3, 2, 4).contiguous()
        log_Z = log_Z.permute(0, 1, 3, 2, 4).contiguous()

        loc_full_shape = list(loc_q.shape)
        log_Z_full_shape = list(log_Z.shape)
        z_full_shape = list(z_logits.shape)
        loc_full_shape[2] = loc_full_shape[2] + len(self.graph_representation.static_nodes)
        log_Z_full_shape[2] = log_Z_full_shape[2] + len(self.graph_representation.static_nodes)
        z_full_shape[2] = z_full_shape[2] + len(self.graph_representation.static_nodes)

        loc_full = torch.zeros(loc_full_shape, device=loc_q.device)
        log_Z_full = torch.zeros(log_Z_full_shape, device=log_Z.device)
        z_logits_full = torch.zeros(z_full_shape, device=z_logits.device)
        loc_full[..., 0] = 1.
        log_Z_full[..., :] = 100.
        z_logits_full[..., 0] = 1.
        loc_full[:, :, self.graph_representation.dynamic_nodes] = loc_q
        log_Z_full[:, :, self.graph_representation.dynamic_nodes] = log_Z
        z_logits_full[:, :, self.graph_representation.dynamic_nodes] = z_logits

        z = torch.distributions.Categorical(logits=z_logits_full)
        loc_q = loc_full
        log_Z = log_Z_full

        M, Z = self.to_bingham_M_Z(loc_q, log_Z)

        dist_q = BinghamMixtureModel(z, M, Z)

        q = GaussianMixtureModel.zero_variance(z, loc_q).mode
        if not self.training:
            p = self.graph_representation.forward_kinematics(q, include_root_rotation=False)  # Careful this is on CPU as of now and slow
        else:
            p = None

        dist_q_chained = ChainedBinghamMixtureModel(z, ChainedBingham(self._chain_list, Bingham(M, Z)))

        return dist_q_chained, {
            'dist_q': dist_q,
            'q': q,
            'p': p,
            'z': z,
            **kwargs
        }

    def hparams(self):
        return {}

    def to_bingham_M_Z(self, loc, log_Z):
        bs = list(loc.shape[:-1])
        M_uns = torch.stack([
            qmul(loc, torch.tensor([0, 1., 0, 0], device=loc.device).expand(bs + [4])),
            qmul(loc, torch.tensor([0, 0, 1., 0], device=loc.device).expand(bs + [4])),
            qmul(loc, torch.tensor([0, 0, 0, 1.], device=loc.device).expand(bs + [4])),
        ], dim=-2)
        log_z_desc, sort_idx = torch.sort(log_Z, dim=-1, descending=True)
        sort_idx = sort_idx.unsqueeze(-1).repeat([1] * (len(bs) + 1) + [4])
        M = M_uns.gather(dim=-2, index=sort_idx)
        M = torch.cat([M, loc.unsqueeze(-2)],
                      dim=-2)
        Z = -torch.sigmoid(
            log_z_desc) * (900. - 1e-3) - 1e-3  # force vanisihing gradient towards the limit so that the model can concentrate on the mean
        #Z = -torch.ones_like(Z) * 700
        Z = torch.cat([Z, torch.zeros(Z.shape[:-1] + (1,), device=Z.device)], dim=-1)
        return M, Z

    def slerp_lp(self, x):
        hrange = 0.4
        hbias = 0.4
        low = max(min(hbias - (hrange / 2), 1), 0)
        high = max(min(hbias + (hrange / 2), 1), 0)
        hrangeLimited = high - low
        ts = x.shape[1]
        xt = x[:, 0]
        for t in range(1, ts):
            xt1 = x[:, t]
            d = qdist(xt, xt1)

            hlpf = (d / np.pi) * hrangeLimited + low
            xt = self.slerp(xt, xt1, hlpf)
            x[:, t] = xt

        return x

    def slerp(self, low, high, value):
        low_norm = low
        high_norm = high
        omega = torch.acos((low_norm * high_norm).sum(dim=-1).clamp(min=-1., max=1.))
        so = torch.sin(omega)
        mask = (so.abs() > 1e-4).type_as(so)
        so = mask * so + (1-mask) * torch.ones_like(so)
        res = (torch.sin((1.0 - value) * omega) / so).unsqueeze(dim=-1) * low + (torch.sin(value * omega) / so).unsqueeze(
            dim=-1) * high
        return res * mask.unsqueeze(-1) + (1-mask.unsqueeze(-1)) * high