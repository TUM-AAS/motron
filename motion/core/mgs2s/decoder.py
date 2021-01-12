import torch
import torch.nn as nn

from common.torch import get_activation_function

# TODO: Idea: Can you use the inverse of the toGMM to populate the first input to the GRU e.g. control->state
from directional_distributions import Bingham
from motion.components.graph_linear import NodeLinear
from motion.components.graph_lstm import GraphLSTM, NodeLSTM, StackedNodeLSTM
from motion.components.structural import GraphLinear, StaticGraphLinear, StaticGraphLSTM, StaticGraphRNN
from motion.components.to_bingham_param import ToBMMParameter
from motion.components.to_gmm_param import ToGMMParameter

def quat_act2(x):
    x = torch.tanh(x)
    return (13./5.)*(torch.pow(0.6*x, 3) - 0.6*x)

class QuatAct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=-1, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < -1] *= -0.01
        grad_input[input > 1] *= -0.01
        return grad_input


class Decoder(nn.Module):
    def __init__(self,
                 graph_influence: torch.nn.Parameter,
                 node_types,
                 feature_size: int,
                 input_size: int,
                 hidden_size: int,
                 latent_size: int,
                 output_size: int,
                 prediction_horizon: int,
                 state_representation,
                 dec_num_layers: int = 1,
                 dec_activation_fn: object = None,
                 param_groups=None,
                 **kwargs):
        super().__init__()
        self.param_groups = param_groups
        self._state_representation = state_representation
        self._prediction_horizon = prediction_horizon
        self.activation_fn = get_activation_function(dec_activation_fn)
        self.rnn = StaticGraphLSTM(feature_size, hidden_size, num_layers=1, graph_influence=graph_influence, learn_influence=True, node_types=node_types, dropout=0.5, recurrent_dropout=0., clockwork=True, learn_additive_graph_influence=True)

        self.fc = StaticGraphLinear(hidden_size, output_size, graph_influence=graph_influence, learn_influence=True, node_types=node_types)
        self.fc2 = StaticGraphLinear(hidden_size, output_size, graph_influence=graph_influence, learn_influence=True, node_types=node_types)
        self.initial_hidden1 = StaticGraphLinear(latent_size + input_size, hidden_size, graph_influence=graph_influence, node_types=node_types)
        self.initial_hidden2 = StaticGraphLinear(latent_size + input_size, hidden_size, graph_influence=graph_influence, node_types=node_types)
        self.dropout = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.2)

        # self.to_gmm_params = ToGMMParameter(output_size // 21,
        #                                     output_state_size=4,
        #                                     dist_state_size=3,
        #                                     **kwargs)

        self.to_bmm_params = ToBMMParameter(graph_influence,
                                            node_types,
                                            output_size,
                                            output_state_size=4,
                                            dist_state_size=4,
                                            **kwargs)

    def forward(self, x: torch.Tensor, enc: torch.Tensor,  z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        nodes = x.shape[2]
        out = []
        loc_d_l = []
        out_log_diag = []
        #u0 = self.initial_input(x)
        # Initialize hidden state of rnn
        loc_start = x[..., :4].clone()

        xi = x
        enc_z = torch.cat([z, enc], dim=-1)
        rnn_h1 = self.initial_hidden1(enc_z)
        rnn_h2 = self.initial_hidden2(enc_z)
        hidden = [(rnn_h1, rnn_h2, None)]
        for i in range(self._prediction_horizon):
            rnn_out, hidden = self.rnn((xi), hidden, i)
            yi = (rnn_out).squeeze(1)
            yi1 = torch.tanh(self.fc(self.dropout(torch.tanh(yi))))
            yi2 = torch.tanh(self.fc2(self.dropout1(torch.tanh(yi))))
            loc, log_Z = self.to_bmm_params(yi1, yi2)
            w = torch.ones(loc.shape[:-1] + (1,), device=loc.device)
            loc = loc
            loc = torch.cat([w, loc], dim=-1)
            loc_d = self._state_representation.validate(loc)
            loc = self._state_representation.sum(loc_start, loc_d)
            out.append(loc)
            loc_d_l.append(loc_d)
            out_log_diag.append(log_Z)
            if self.param_groups[0]['teacher_forcing_factor'] > 1e-6 and self.training and y is not None:
                teacher_forcing_mask = (torch.rand(list(loc.shape[:-2]) + [1] * 2)
                                        < self.param_groups[0]['teacher_forcing_factor'])
                loc_start = teacher_forcing_mask.type_as(y) * y[:, i] + (~teacher_forcing_mask).type_as(y) * loc
            else:
                loc_start = loc
            xi = torch.cat([loc_start, loc_d], dim=-1).unsqueeze(1)
        return torch.stack(out, dim=1).contiguous(), torch.stack(loc_d_l, dim=1).contiguous(), torch.stack(out_log_diag, dim=1).contiguous()
