import torch
import pytest
from transformer.components.decoder import Decoder
from transformer.components.feed_forward import PositionwiseFeedForward
from transformer.components.layer_norm import LayerNorm
from transformer.components.multi_head import MultiHeadAttention


@pytest.mark.parametrize(
    "batch_size,src_seq, tgt_seq, d_model, d_ff, num_heads",
    [(2, 16, 12, 32, 64, 4), (5, 8, 8, 16, 32, 2)],
)
def test_forward_shape(batch_size, src_seq, tgt_seq, d_model, d_ff, num_heads):
    src = torch.randn(batch_size, src_seq, d_model)
    kv = torch.randn(batch_size, src_seq, d_model)
    tgt = torch.randn(batch_size, tgt_seq, d_model)
    tgt_mask = torch.tril(torch.ones(tgt_seq, tgt_seq))
    decoder = Decoder(num_layers=3,d_model=d_model,d_ff=d_ff, num_heads=num_heads, attention_cls=MultiHeadAttention, feedforward_cls=PositionwiseFeedForward, norm_cls=LayerNorm)
    decoder_out = decoder(tgt, kv, tgt_mask)

    assert decoder_out.shape == tgt.shape
