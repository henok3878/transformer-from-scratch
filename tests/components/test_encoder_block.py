import torch
from transformer.components.encoder_block import EncoderBlock


def test_forward_shape():
    batch_size, seq_len, d_model, d_ff, num_heads = 4, 32, 16, 32, 4
    encoder_block = EncoderBlock(d_model, d_ff, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    encoder_output = encoder_block(x)

    assert x.shape == encoder_output.shape
