import torch
import pytest
from transformer.components.encoder import Encoder


@pytest.mark.parametrize(
    "batch_size,src_seq, d_model, d_ff, num_heads",
    [(2, 16, 32, 64, 4), (5, 8, 16, 32, 2)],
)
def test_forward_shape(batch_size, src_seq, d_model, d_ff, num_heads):
    src = torch.randn(batch_size, src_seq, d_model)
    encoder = Encoder(3, d_model, d_ff, num_heads)
    encoder_out = encoder(src)

    assert encoder_out.shape == src.shape
