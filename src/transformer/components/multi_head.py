import math
import torch
import torch.nn as nn
from typing import Union
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads  # h in the paper
        assert d_model % num_heads == 0, "Make sure that d_model % num_heads == 0"
        self.head_dim = d_model // num_heads  # dk, dq, and dv in the paper

        # q, k, v projections are concatnation of each head's projection
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, kv: torch.Tensor, mask: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        """
        Args:
            x: shape = (batch_size, x_seq_len, d_model)
            kv: shape = (batch_size, kv_seq_len, d_model)
        """

        query = self.q_proj(x)  # shape: (batch_size, x_seq_len, d_model)
        key = self.k_proj(kv)  # shape: (batch_size, kv_seq_len, d_model)
        value = self.v_proj(kv)  # shape: (batch_size,kv_seq_len , d_model)

        # reshape to heads
        batch_size, x_seq_len, _ = x.shape
        _, kv_seq_len, _ = kv.shape

        query = query.view(
            batch_size, x_seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        value = value.view(
            batch_size, kv_seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # score = Q@K^T
        # Q shape = (batch_size, num_heads, x_seq_len, head_dim)
        # K shpape = (batch_size, num_heads, kv_seq_len, head_dim)
        # K.transpose(-2, -1) = (batch_size, num_heads, head_dim, kv_seq_len)
        # scores shape: (batch_size, num_heads, x_seq_len,kv_seq_len)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        # shape = (batch_size, num_heads, x_seq_len, kv_seq_len)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # attn shape: (batch_size, num_heads, x_seq_len, kv_seq_len)
        # value shape: (batch_size, num_heads, kv_seq_len, head_dim)
        # context shape: (batch_size, num_heads, x_seq_len, head_dim)
        context = attn @ value

        # concatnation step
        # expected shape: (batch_size, x_seq_len, d_model) to get this
        # first we gotta transpose 1st and 2nd dims
        # context.transpose(1,2) shape = (batch_size, x_seq_len, num_heads,head_dim)
        # since transpose makes the tensor non contiguous, we need to use contiguous before view
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, x_seq_len, self.d_model)
        )

        output = self.out_proj(context)
        return output
