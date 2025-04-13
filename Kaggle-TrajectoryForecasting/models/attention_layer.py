from typing import Optional
import math

import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings
from torch.nn.parameter import Parameter
import torch
from torch.nn.init import xavier_uniform_, constant_
from torch import Tensor
import torch.nn.functional as F
from torch.nn.functional import linear


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config['hidden_dim']
        self.linear1 = nn.Linear(config["embed_dim"], hidden_dim * 2)
        self.linear2 = nn.Linear(hidden_dim, config['embed_dim'])
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.linear1(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = self.act(x1) * x2
        return self.linear2(x)


# https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
# https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/activation.py#L973
class MultiheadAttentionWithRoPE(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_head: int,
            dropout: float = 0.0,
            bias: bool = True,
            ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.dropout = dropout
        self.head_dim = embed_dim // num_head
        assert (
                self.head_dim * num_head == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)

        # https://arxiv.org/abs/2104.09864
        self.rotary_pos_enc = RotaryPositionalEmbeddings(self.head_dim, max_seq_len=50)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
    ):
        is_batched = query.dim() == 3

        if not is_batched:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(0)

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        N, L, E = query.shape

        w_q, w_k, w_v = self.in_proj_weight.chunk(3)
        if self.in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = self.in_proj_bias.chunk(3)

        # (N, L, E)
        q, k, v = linear(query, w_q, b_q), linear(key, w_k, b_k), linear(value, w_v, b_v)

        # rotary positional encoding
        q = self.rotary_pos_enc(q.reshape(N, L, self.num_head, self.head_dim)).reshape(N, L, E)
        k = self.rotary_pos_enc(k.reshape(N, L, self.num_head, self.head_dim)).reshape(N, L, E)

        # (N, L, E) -> (L, N, E)
        q, k, v = q.transpose(0, 1).contiguous(), k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous()

        # (L, N, E) -> (N * num_head, L, head_dim)
        q = q.view(L, N * self.num_head, self.head_dim).transpose(0, 1)
        k = k.view(L, N * self.num_head, self.head_dim).transpose(0, 1)
        v = v.view(L, N * self.num_head, self.head_dim).transpose(0, 1)

        if need_weights:
            E = q.shape[-1]
            q_scaled = q * math.sqrt(1.0 / float(E))
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            if attn_mask is not None:
                attn_output_weights += attn_mask
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)

            if self.training:
                attn_output_weights = F.dropout(attn_output_weights, p=self.dropout)

            attn_output = torch.bmm(attn_output_weights, v)  # (N * num_head, L, head_dim)

            # (N * num_head, L, head_dim) -> (N, L, embed_dim)
            attn_output = attn_output.transpose(0, 1).contiguous().view(L, N, self.embed_dim).transpose(0, 1)
            attn_output = self.out_proj(attn_output)

            # (N * num_head, L, L) -> (N, num_head, L, L)
            attn_output_weights = attn_output_weights.view(N, self.num_head, L, L)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)

            if not is_batched:
                attn_output = attn_output.squeeze(0)
                attn_output_weights = attn_output_weights.squeeze(0)
            return attn_output, attn_output_weights

        else:
            if attn_mask is not None:
                # attn_mask can be either (L,S) or (N * num_heads, L, S)
                # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
                # in order to match the input for SDPA of (N, num_heads, L, S)
                if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(0)
                else:
                    attn_mask = attn_mask.view(N, self.num_head, -1, L)

            q = q.view(N, self.num_head, L, self.head_dim)
            k = k.view(N, self.num_head, L, self.head_dim)
            v = v.view(N, self.num_head, L, self.head_dim)

            # (N, num_head, L, head_dim) -> (N, L, E)
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=self.dropout if self.training else 0.0)
            attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(N, L, E)
            attn_output = self.out_proj(attn_output)

            if not is_batched:
                attn_output = attn_output.squeeze(0)
            return attn_output, None


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config['embed_dim']
        self.num_head = config['num_head']
        self.hidden_dim = config['hidden_dim']
        self.dropout = config['dropout']

        if config['use_rope']:
            self.temporal_attention = MultiheadAttentionWithRoPE(self.embed_dim, self.num_head, self.dropout)
        else:
            self.temporal_attention = nn.MultiheadAttention(self.embed_dim, self.num_head, dropout=self.dropout, batch_first=True)

        self.social_attention = nn.MultiheadAttention(self.embed_dim, self.num_head, dropout=self.dropout, batch_first=True)

        self.norm1 = nn.RMSNorm(normalized_shape=self.embed_dim)
        self.norm2 = nn.RMSNorm(normalized_shape=self.embed_dim)
        self.norm3 = nn.RMSNorm(normalized_shape=self.embed_dim)

        self.mlp = SwiGLU(config)

        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        self.dropout3 = nn.Dropout(self.dropout)

    def forward(self, x, temporal_mask=None, social_mask=None):
        """
        :param x: (N, A, T, D)
        :param temporal_mask:
        :param social_mask:
        :return:
        """
        N, A, T, D = x.shape

        # attend on time
        x = x.reshape(N * A, T, D)
        norm_x = self.norm1(x)
        x = x + self.dropout1(self.temporal_attention(norm_x, norm_x, norm_x, attn_mask=temporal_mask)[0])

        # attend on space
        x = x.reshape(N, A, T, D).transpose(1, 2).reshape(N * T, A, D)
        norm_x = self.norm2(x)
        x = x + self.dropout2(self.social_attention(norm_x, norm_x, norm_x, attn_mask=social_mask)[0])

        x = x.reshape(N, T, A, D).transpose(1, 2)
        x = x + self.dropout3(self.mlp(self.norm3(x)))

        return x
