import torch
import torch.nn as nn
from transformer.components.layer_norm import LayerNorm
from transformer.components.multi_head import MultiHeadAttention
from transformer.components.feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, source_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # self attn
        attn_out = self.self_attn(x, x, mask=source_mask)
        # residual + norm for attn sublayer
        x = self.norm_attn(x + self.dropout_attn(attn_out))

        # position wise feed forward
        ffn_out = self.ffn(x)
        # residual + norm for ffn sublayer
        x = self.norm_ffn(x + self.dropout_ffn(ffn_out))

        return x
