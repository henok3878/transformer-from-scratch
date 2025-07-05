import torch
from transformer.components.decoder_layer import DecoderLayer


def test_forward_shape():
    batch_size, seq_len, d_model, d_ff, num_heads = 4, 32, 16, 32, 4
    decoder_layer = DecoderLayer(d_model, d_ff, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    decoder_output = decoder_layer(x, x)

    assert x.shape == decoder_output.shape


def test_forward_shape_with_mask():
    batch_size, src_seq_len, tgt_seq_len, d_model, d_ff, num_heads = (
        4,
        64,
        48,
        16,
        32,
        4,
    )

    # letting pytorch handle the actual mask shape via braodcasting
    mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len))
    decoder_layer = DecoderLayer(d_model, d_ff, num_heads)
    # the target sequence
    x = torch.randn(batch_size, tgt_seq_len, d_model)
    # output of the encoder
    kv = torch.randn(batch_size, src_seq_len, d_model)
    decoder_output = decoder_layer(x, kv, target_mask=mask)

    assert x.shape == decoder_output.shape and decoder_output.dtype == x.dtype
