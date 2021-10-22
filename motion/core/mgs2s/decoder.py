from typing import Tuple, Union

import torch
import torch.nn as nn
from motion.quaternion import Quaternion

from motion.components.structural import StaticGraphLinear, StaticGraphGRU


class Decoder(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 feature_size: int,
                 input_size: int,
                 hidden_size: int,
                 latent_size: int,
                 output_size: int,
                 position: bool,
                 G: Union[torch.Tensor, torch.nn.Parameter] = None,
                 T: torch.Tensor = None,
                 dec_num_layers: int = 1,
                 dropout: float = 0.,
                 param_groups=None,
                 **kwargs):
        super().__init__()
        self.position = position

        self.param_groups = param_groups
        self.activation_fn = torch.tanh
        self.num_layers = dec_num_layers

        self.initial_hidden_h = StaticGraphLinear(latent_size + input_size + feature_size,
                                                  hidden_size,
                                                  num_nodes=num_nodes,
                                                  learn_influence=True,
                                                  node_types=T)

        self.rnn = StaticGraphGRU(feature_size + latent_size + input_size,
                                  hidden_size,
                                  num_nodes=num_nodes,
                                  num_layers=dec_num_layers,
                                  learn_influence=True,
                                  node_types=T,
                                  recurrent_dropout=dropout,
                                  learn_additive_graph_influence=True,
                                  clockwork=False)

        self.fc_q = StaticGraphLinear(hidden_size,
                                      output_size,
                                      num_nodes=num_nodes,
                                      learn_influence=True,
                                      node_types=T)

        self.fc_cov_lat = StaticGraphLinear(hidden_size,
                                            output_size,
                                            num_nodes=num_nodes,
                                            learn_influence=True,
                                            node_types=T)

        if position:
            self.to_q = StaticGraphLinear(output_size, 3, num_nodes=num_nodes-1, node_types=T[:-1])
            self.to_q_cov_lat = StaticGraphLinear(output_size, 6, num_nodes=num_nodes-1, node_types=T[:-1])

            self.to_p = torch.nn.Linear(output_size, 3)
            self.to_p_cov_lat = torch.nn.Linear(output_size, 6)
        else:
            self.to_q = StaticGraphLinear(output_size, 3, num_nodes=num_nodes, node_types=T)
            self.to_q_cov_lat = StaticGraphLinear(output_size, 6, num_nodes=num_nodes, node_types=T)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, h: torch.Tensor, z: torch.Tensor, q_t: torch.Tensor, p_t: torch.Tensor = None,
                ph: int = 1, state=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dq = list()
        dq_cov_lat = list()

        if self.position:
            dp = list()
            dp_cov_lat = list()

        x_t = x[:, -1]
        x_t_s = x_t.clone()
        if state is None:
            x_t_1 = x[:, -2]
        else:
            x_t_1 = state

        h_z = torch.cat([z, h], dim=-1)

        # Initialize hidden state of rnn
        rnn_h = self.initial_hidden_h(torch.cat([x_t_1, h_z], dim=-1))
        hidden = [(rnn_h, None)] * self.num_layers

        for i in range(ph):
            # Run RNN
            rnn_out, hidden = self.rnn(torch.cat([x_t, h_z], dim=-1).unsqueeze(1), hidden, i)  # [B * Z, 1, N, D]
            y_t = rnn_out.squeeze(1)  # [B * Z, N, D]
            y_t = self.dropout(self.activation_fn(y_t))

            y_t_state = self.fc_q(y_t)
            y_t_cov_lat = self.fc_cov_lat(y_t)

            y_t_state = self.activation_fn(y_t_state)
            y_t_cov_lat = torch.tanh(y_t_cov_lat)

            if self.position:
                dq_t_3 = self.to_q(y_t_state[..., :-1, :])
                cov_q_lat_t = self.to_q_cov_lat(y_t_cov_lat[..., :-1, :])
                dq_t = Quaternion(angle=torch.norm(dq_t_3, dim=-1), axis=dq_t_3).q
                q_t = Quaternion.mul_(dq_t, q_t)

                dp_t = self.to_p(y_t_state[..., -1, :]) / 20
                cov_p_lat_t = self.to_p_cov_lat(y_t_cov_lat[..., -1, :])

                p_t = p_t + dp_t

                dp.append(dp_t)
                dp_cov_lat.append(cov_p_lat_t)
                dp_padded = torch.nn.functional.pad(dp_t, (0, 5)).unsqueeze(1)
                x_t = torch.cat([torch.cat([q_t, dq_t], dim=-1),  dp_padded], dim=1)
            else:
                dq_t_3 = self.to_q(y_t_state)
                cov_q_lat_t = self.to_q_cov_lat(y_t_cov_lat)
                dq_t = Quaternion(angle=torch.norm(dq_t_3, dim=-1), axis=dq_t_3).q
                q_t = Quaternion.mul_(dq_t, q_t)
                x_t = torch.cat([q_t, dq_t], dim=-1)

            dq.append(dq_t)
            dq_cov_lat.append(cov_q_lat_t)

        dq = torch.stack(dq, dim=1)
        dq_cov_lat = torch.stack(dq_cov_lat, dim=1)

        if self.position:
            dp = torch.stack(dp, dim=1)
            dp_cov_lat = torch.stack(dp_cov_lat, dim=1)
        else:
            dp = None
            dp_cov_lat = None

        return dq, dq_cov_lat, dp, dp_cov_lat, x_t_s
