import torch
import torch.nn as nn
from typing import Type
from transformer.components.base import attention
from transformer.components.base.attention import BaseAttention
from transformer.components.base.feed_forward import BaseFeedForward
from transformer.components.layer_norm import LayerNorm
from transformer.components.multi_head import MultiHeadAttention
from transformer.components.feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, attention_cls:Type[BaseAttention], feedforward_cls: Type[BaseFeedForward], norm_cls: Type[nn.Module], dropout: float=0.1, **kwargs):
        super().__init__()

        self.self_attn = attention_cls(d_model, num_heads, dropout, **kwargs)
        self.dropout_self_attn = nn.Dropout(dropout)
        self.norm_self_attn = norm_cls(d_model, **kwargs)

        self.cross_attn = attention_cls(d_model, num_heads, dropout, **kwargs)
        self.dropout_cross_attn = nn.Dropout(dropout)
        self.norm_cross_attn = norm_cls(d_model, **kwargs)

        self.ffn = feedforward_cls(d_model, d_ff, dropout, **kwargs)
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = norm_cls(d_model, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attn = self.self_attn(x, x, mask=target_mask)
        x = self.norm_self_attn(x + self.dropout_self_attn(self_attn))

        cross_attn = self.cross_attn(x, kv, mask=kv_mask)
        x = self.norm_cross_attn(x + self.dropout_cross_attn(cross_attn))

        ffn_out = self.ffn(x)
        x = self.norm_ffn(x + self.dropout_ffn(ffn_out))

        return x
