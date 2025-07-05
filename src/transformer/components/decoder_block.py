import torch
import torch.nn as nn
from transformer.components.layer_norm import LayerNorm
from transformer.components.multi_head import MultiHeadAttention
from transformer.components.feed_forward import PositionwiseFeedForward


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.layer_norm3 = LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attn = self.self_attn(x, x, mask=target_mask)
        x = self.layer_norm1(x + self.dropout1(self_attn))

        cross_attn = self.cross_attn(x, kv, mask=kv_mask)
        x = self.layer_norm2(x + self.dropout2(cross_attn))

        ffn_output = self.feed_forward(x)
        x = self.layer_norm3(x + self.dropout3(ffn_output))

        return x
