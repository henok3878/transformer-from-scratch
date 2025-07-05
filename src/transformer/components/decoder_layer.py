import torch
import torch.nn as nn
from transformer.components.layer_norm import LayerNorm
from transformer.components.multi_head import MultiHeadAttention
from transformer.components.feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout_self_attn = nn.Dropout(dropout)
        self.norm_self_attn = LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout_cross_attn = nn.Dropout(dropout)
        self.norm_cross_attn = LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = LayerNorm(d_model)

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
