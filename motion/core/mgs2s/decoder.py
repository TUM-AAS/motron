from typing import Tuple, Union

import torch
import torch.nn as nn

from motion.bingham import Bingham
from motion.components.to_bingham import ToBingham
from motion.quaternion import Quaternion

from motion.components.structural import StaticGraphLinear, StaticGraphLSTM
from motion.utils.torch import get_activation_function


class Decoder(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 feature_size: int,
                 input_size: int,
                 hidden_size: int,
                 latent_size: int,
                 output_size: int,
                 G: Union[torch.Tensor, torch.nn.Parameter] = None,
                 T: torch.Tensor = None,
                 dec_num_layers: int = 1,
                 dropout: float = 0.,
                 param_groups=None,
                 **kwargs):
        super().__init__()
        self.param_groups = param_groups
        self.activation_fn = torch.tanh
        self.num_layers = dec_num_layers

        self.initial_hidden_c = StaticGraphLinear(latent_size + input_size,
                                                  hidden_size,
                                                  num_nodes=num_nodes,
                                                  learn_influence=True,
                                                  node_types=T)

        self.initial_hidden_h = StaticGraphLinear(feature_size,
                                                  hidden_size,
                                                  num_nodes=num_nodes,
                                                  node_types=T)

        self.rnn = StaticGraphLSTM(feature_size + latent_size + input_size,
                                   hidden_size,
                                   num_nodes=num_nodes,
                                   num_layers=dec_num_layers,
                                   learn_influence=True,
                                   node_types=T,
                                   recurrent_dropout=0.1,
                                   learn_additive_graph_influence=True)

        self.fc_q = StaticGraphLinear(hidden_size,
                                      output_size,
                                      num_nodes=num_nodes,
                                      learn_influence=False,
                                      node_types=T)

        self.fc_Z = StaticGraphLinear(hidden_size,
                                      output_size,
                                      num_nodes=num_nodes,
                                      learn_influence=False,
                                      node_types=T)

        self.ln_q = nn.LayerNorm(output_size, elementwise_affine=False)
        self.ln_Z = nn.LayerNorm(output_size, elementwise_affine=False)

        self.to_q = StaticGraphLinear(output_size, 3, num_nodes=num_nodes, node_types=T)
        self.to_Z = StaticGraphLinear(output_size, 6, num_nodes=num_nodes, node_types=T)

        self.dropout = nn.Dropout(dropout)

        #self.to_bingham = ToBingham(1950.)

    def forward(self, x: torch.Tensor, h: torch.Tensor, z: torch.Tensor, q_t: torch.Tensor, y: torch.Tensor, ph: int = 1) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        q = list()
        Z = list()
        exp_map = list()

        x_t = x[:, -1]
        x_t_1 = x[:, -2]
        h_z = torch.cat([z, h], dim=-1)

        # Initialize hidden state of rnn
        rnn_h = self.initial_hidden_h(x_t_1)
        rnn_c = self.initial_hidden_c(h_z)
        hidden = [(rnn_h, rnn_c, None)] * self.num_layers

        z_t = torch.ones(q_t.shape[:-1] + (6,), device=q_t.device) * 1e-3

        for i in range(ph):
            # Run LSTM
            rnn_out, hidden = self.rnn(torch.cat([x_t, h_z], dim=-1).unsqueeze(1), hidden, i)  # [B * Z, 1, N, D]
            y_t = rnn_out.squeeze(1)  # [B * Z, N, D]
            y_t = self.dropout(self.activation_fn(y_t))

            y_t_q = self.fc_q(y_t)
            y_t_Z = self.fc_Z(y_t)

            y_t_q = self.activation_fn(y_t_q)
            y_t_Z = torch.tanh(y_t_Z)

            y_t_Z = self.ln_Z(y_t_Z)

            dq_t_3 = self.to_q(y_t_q)
            Z_t = torch.sigmoid(self.to_Z(y_t_Z))

            exp_map.append(dq_t_3)
            dq_t = Quaternion(angle=torch.norm(dq_t_3, dim=-1), axis=dq_t_3).q#Quaternion(torch.cat([w, dq_t_3], dim=-1)).normalized.q #

            q_t = Quaternion.mul_(dq_t, q_t)  # Quaternion multiplication
            z_t = z_t + Z_t

            q.append(q_t)
            Z.append(z_t)
            if self.param_groups[0]['teacher_forcing_factor'] > 1e-6 and self.training and y is not None:
                teacher_forcing_mask = (torch.rand(list(q_t.shape[:-2]) + [1] * 2)
                                        < self.param_groups[0]['teacher_forcing_factor'])
                q_t = teacher_forcing_mask.type_as(y) * y[:, i] + (~teacher_forcing_mask).type_as(y) * q_t

            # if self.training:
            #     M, Z_tb = self.to_bingham(q_t, Z_t)
            #     q_t = Bingham(M, Z_tb).sample()

            #q_t_scaled = q_t[..., 1:] / ((1+1e-1) - q_t[..., [0]])
            #dq_t_scaled = dq_t[..., 1:] / ((1 + 1e-1) - dq_t[..., [0]])
            x_t = torch.cat([q_t.detach(), dq_t.detach()], dim=-1)


        q = torch.stack(q, dim=1)
        Z = torch.stack(Z, dim=1)
        exp_map = torch.stack(exp_map, dim=1)
        return q, Z, {'exp_map': exp_map}
