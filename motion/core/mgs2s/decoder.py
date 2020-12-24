import torch
import torch.nn as nn

from common.torch import get_activation_function

# TODO: Idea: Can you use the inverse of the toGMM to populate the first input to the GRU e.g. control->state
from directional_distributions import Bingham
from motion.components.graph_linear import GraphLinear, NodeLinear
from motion.components.graph_lstm import GraphLSTM, NodeLSTM, StackedNodeLSTM
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
        self.rnn = StackedNodeLSTM([{'graph_influence':graph_influence, 'node_types':node_types, 'input_size':latent_size + input_size, 'hidden_size':output_size, 'node_dropout':0.0, 'recurrent_dropout': 0.5},
                                    {'graph_influence':graph_influence, 'node_types':node_types, 'input_size':output_size, 'hidden_size':output_size, 'node_dropout':0.3, 'recurrent_dropout': 0.5}], dropout=0.3)
        self.rnn2 = StackedNodeLSTM([{'graph_influence': graph_influence, 'node_types': node_types,
                                     'input_size': latent_size + input_size, 'hidden_size': output_size,
                                     'node_dropout': 0.0, 'recurrent_dropout': 0.5},
                                    {'graph_influence': graph_influence, 'node_types': node_types,
                                     'input_size': output_size, 'hidden_size': output_size, 'node_dropout': 0.3,
                                     'recurrent_dropout': 0.5}], dropout=0.3)
        self.fc = NodeLinear(graph_influence=graph_influence, in_features=output_size, out_features=output_size)
        self.fc2 = NodeLinear(graph_influence=graph_influence, in_features=output_size, out_features=output_size)
        self.initial_hidden1 = GraphLinear(graph_influence=graph_influence, in_features=hidden_size, out_features=output_size)
        self.initial_hidden2 = GraphLinear(graph_influence=graph_influence, in_features=hidden_size,
                                           out_features=output_size)
        self.dropout = nn.Dropout(0.5)

        # self.to_gmm_params = ToGMMParameter(output_size // 21,
        #                                     output_state_size=4,
        #                                     dist_state_size=3,
        #                                     **kwargs)

        self.to_bmm_params = ToBMMParameter(graph_influence,
                                            output_size,
                                            output_state_size=4,
                                            dist_state_size=4,
                                            **kwargs)

    def forward(self, x: torch.Tensor, enc: torch.Tensor,  z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        nodes = x.shape[2]
        out = []
        out_log_diag = []
        #u0 = self.initial_input(x)
        # Initialize hidden state of rnn
        loc_start = x[..., :4].clone()

        xi = torch.cat([z, x], dim=-1).view(-1, 1, x.shape[-2], z.shape[-1] + x.shape[-1])
        xi2 = torch.cat([z, x.detach()], dim=-1).view(-1, 1, x.shape[-2], z.shape[-1] + x.shape[-1])
        enc = enc.view(-1, enc.shape[-2], enc.shape[-1])
        rnn_h1 = self.initial_hidden1(enc.squeeze(1))
        rnn_h2 = self.initial_hidden2(enc.squeeze(1))
        hidden = [(rnn_h1, rnn_h2), (rnn_h1, rnn_h2)]
        hidden2 = [(rnn_h1.detach(), rnn_h2.detach()), (rnn_h1.detach(), rnn_h2.detach())]
        G = [torch.Tensor(), torch.Tensor()]
        G2 = [torch.Tensor(), torch.Tensor()]
        for i in range(self._prediction_horizon):
            rnn_out, hidden, G = self.rnn((xi), hidden, G)
            #rnn_out2, hidden2, G2 = self.rnn2((xi2), hidden2, G2)
            yi = (rnn_out).view(bs, -1, nodes, rnn_out.shape[-1])
            yi1 = torch.tanh(self.fc(self.dropout(torch.tanh(yi))))
            #yi2 = (rnn_out2).view(bs, -1, nodes, rnn_out2.shape[-1])
            yi2 = torch.tanh(self.fc2(self.dropout(torch.tanh(yi))))
            loc, log_Z = self.to_bmm_params(yi1, yi2)
            w = torch.ones(loc.shape[:-1] + (1,), device=loc.device)
            loc = torch.cat([w, loc], dim=-1)
            loc_d = self._state_representation.validate(loc)
            loc = self._state_representation.sum(loc_start, loc_d)
            out.append(loc)
            out_log_diag.append(log_Z)
            if self.training and y is not None:
                teacher_forcing_mask = (torch.rand(list(loc.shape[:-2]) + [1] * 2)
                                        < self.param_groups[0]['teacher_forcing_factor'])
                loc_start = teacher_forcing_mask.type_as(y) * y[:, [i]] + (~teacher_forcing_mask).type_as(y) * loc
            else:
                loc_start = loc
            xi = torch.cat([z, loc_start, loc_d], dim=-1).view(-1, 1, x.shape[-2], z.shape[-1] + x.shape[-1])
            xi2 = torch.cat([z, loc_start.detach(), loc_d.detach()], dim=-1).view(-1, 1, x.shape[-2], z.shape[-1] + x.shape[-1])
        return torch.stack(out, dim=1).contiguous(), torch.stack(out_log_diag, dim=1).contiguous()
