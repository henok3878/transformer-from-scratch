import torch
import torch.nn as nn
from transformer.components.layer_norm import LayerNorm
from transformer.components.multi_head import MultiHeadAttention
from transformer.components.feed_forward import PositionwiseFeedForward


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self attn
        attn_output = self.self_attn(x, x)
        # residual + norm for attn sublayer
        x = self.layer_norm1(x + self.dropout1(attn_output))

        # position wise feed forward
        ffn_output = self.feed_forward(x)
        # residual + norm for ffn sublayer
        x = self.layer_norm2(x + self.dropout2(ffn_output))

        return x
