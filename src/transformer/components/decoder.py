import torch
import torch.nn as nn
from transformer.components.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, kv, target_mask, kv_mask)
        return x
