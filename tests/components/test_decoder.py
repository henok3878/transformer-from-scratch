import torch
import pytest
from transformer.components.decoder import Decoder


@pytest.mark.parametrize(
    "batch_size,src_seq, tgt_seq, d_model, d_ff, num_heads",
    [(2, 16, 12, 32, 64, 4), (5, 8, 8, 16, 32, 2)],
)
def test_forward_shape(batch_size, src_seq, tgt_seq, d_model, d_ff, num_heads):
    src = torch.randn(batch_size, src_seq, d_model)
    kv = torch.randn(batch_size, src_seq, d_model)
    tgt = torch.randn(batch_size, tgt_seq, d_model)
    tgt_mask = torch.tril(torch.ones(tgt_seq, tgt_seq))
    decoder = Decoder(3, d_model, d_ff, num_heads)
    decoder_out = decoder(tgt, kv, tgt_mask)

    assert decoder_out.shape == tgt.shape
