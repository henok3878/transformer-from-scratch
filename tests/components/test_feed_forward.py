import torch
from transformer.components.feed_forward import PositionwiseFeedForward


def test_forward_shape():
    batch_size, seq_len, d_model, d_ff = 8, 16, 16, 32
    x = torch.randn(batch_size, seq_len, d_model)
    feed_forward = PositionwiseFeedForward(d_model, d_ff)
    output = feed_forward(x)

    assert x.shape == output.shape
