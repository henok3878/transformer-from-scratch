import torch
import torch.nn as nn
from typing import Type 
from transformer.components.base.attention import BaseAttention
from transformer.components.base.feed_forward import BaseFeedForward
from transformer.components.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        attention_cls: Type[BaseAttention], 
        feedforward_cls: Type[BaseFeedForward], 
        norm_cls: Type[nn.Module], 
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    dropout=dropout, attention_cls=attention_cls, feedforward_cls=feedforward_cls, norm_cls=norm_cls, 
                    **kwargs
                ) 
                for _ in range(num_layers)
            ]
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
