from typing import Tuple, Union

import torch
import torch.nn as nn

from motion.quaternion import Quaternion

from motion.components.structural import StaticGraphLinear, StaticGraphLSTM
from motion.utils.torch import get_activation_function


class FCDecoder(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 feature_size: int,
                 input_size: int,
                 hidden_size: int,
                 latent_size: int,
                 output_size: int,
                 prediction_horizon: int = 25,
                 G: Union[torch.Tensor, torch.nn.Parameter] = None,
                 T: torch.Tensor = None,
                 dec_num_layers: int = 1,
                 activation_fn: object = None,
                 dropout: float = 0.,
                 param_groups=None,
                 **kwargs):
        super().__init__()
        self.param_groups = param_groups
        self.activation_fn = get_activation_function(activation_fn)
        self.activation_fn = torch.nn.LeakyReLU()
        self.num_layers = dec_num_layers
        self.ph = prediction_horizon


        self.fc1 = StaticGraphLinear(input_size,
                                                  hidden_size,
                                     graph_influence=G,
                                                  num_nodes=num_nodes,
                                                    learn_influence=True,
                                                  node_types=T)

        layers = []
        for i in range(dec_num_layers):
            layers.append(StaticGraphLinear(hidden_size,
                                                  hidden_size,
                                            graph_influence=G,
                                                  num_nodes=num_nodes,
                                                    learn_influence=True,
                                                  node_types=T))

        self.fc2 = StaticGraphLinear(hidden_size,
                                        output_size,
                                     graph_influence=G,
                                        num_nodes=num_nodes,
                                        learn_influence=True,
                                        node_types=T)

        self.layers = torch.nn.ModuleList(layers)

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

        self.to_q = StaticGraphLinear(output_size, 3 * prediction_horizon, num_nodes=num_nodes, node_types=T)
        self.to_Z = StaticGraphLinear(output_size, 3 * prediction_horizon, num_nodes=num_nodes, node_types=T)

        self.dropout = nn.Dropout(0.)

    def forward(self, x: torch.Tensor, h: torch.Tensor,  z: torch.Tensor, q_t, y: torch.Tensor, ph=1) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        q = list()
        Z = list()

        #x_t = x[:, -1]
        #x_t_1 = x[:, -2]
        #h_z = torch.cat([z, h], dim=-1)

        x_input = torch.cat([h], dim=-1)

        y = self.fc1(x_input)
        for layer in self.layers:
            y = y + layer(y)
            y = torch.relu(y)
            y = self.dropout(y)

        y = self.fc2(y)
        y = torch.relu(y)

        dq_3 = self.to_q(y)
        dq_3 = dq_3.view(dq_3.shape[:-1] + (self.ph, 3)).permute(0, 2, 1, 3)
        # Z = self.to_Z(y)
        # Z = Z.view(Z.shape[:-1] + (self.ph, 3)).permute(0, 2, 1, 3)
        #
        # w = torch.ones(dq_3.shape[:-1] + (1,), device=q_t.device)
        #
        # dq_t = Quaternion(angle=torch.norm(dq_3, dim=-1), axis=dq_3).q#Quaternion(torch.cat([w, dq_3], dim=-1)).normalized.q
        #
        # q = dq_t#Quaternion.mul_(dq_t, q_t.unsqueeze(1))  # Quaternion multiplication

        return dq_3
