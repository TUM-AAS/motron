from torch import *
from torch.nn import *
from torch.nn.functional import linear
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.modules.transformer import Transformer, _get_clones


class StaticGraphTransformer(Module):
    def __init__(self, encoder_layer, num_layers):
        super(StaticGraphTransformer, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, query, key, value) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        for mod in self.layers:
            query = mod(query, key, value)

        return query


class StaticGraphTransformerLayer(Module):
    def __init__(self, features, nhead, num_nodes: int = None, graph_influence: Union[torch.Tensor, Parameter] = None,
                 dropout=0.1, activation="relu", kdim=None, vdim=None):
        super(StaticGraphTransformerLayer, self).__init__()
        self.self_attn = GraphMultiheadAttention(features, nhead, dropout=dropout, kdim=kdim, vdim=vdim)


    def forward(self, query, key, value) -> Tensor:
        query = self.self_attn(query, key, value)[0]
        return query


class GraphMultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(GraphMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value):
        bs, ts, ns, ds = query.size()
        query = query.permute(1, 0, 2, 3)
        key = key.permute(1, 0, 2, 3)
        value = value.permute(1, 0, 2, 3)

        head_dim = self.embed_dim // self.num_heads
        assert head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        q = self.embed_q(query)
        k = self.embed_k(key)
        v = value
        q = q * scaling

        q = q.contiguous().view(ts, bs * ns * self.num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(ts, bs * ns * self.num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        attn_output_weights = softmax(
            attn_output_weights, dim=-1)
        attn_output_weights = dropout(attn_output_weights, p=self.dropout, training=training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None