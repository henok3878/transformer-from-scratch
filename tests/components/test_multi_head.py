import torch
from transformer.components.multi_head import MultiHeadAttention


def test_forward_self_shape():
    batch_size, x_seq_len, d_model, num_heads = 4, 16, 8, 2
    x = torch.randn(batch_size, x_seq_len, d_model)
    multi_head = MultiHeadAttention(d_model, num_heads)

    output = multi_head(x, x)
    assert x.shape == output.shape


def test_forward_cross_shape():
    batch_size, x_seq_len, kv_seq_len, d_model, num_heads = 4, 16, 16, 8, 2
    x = torch.randn(batch_size, x_seq_len, d_model)
    kv = torch.randn(batch_size, kv_seq_len, d_model)
    multi_head = MultiHeadAttention(d_model, num_heads)

    output = multi_head(x, kv)

    assert x.shape == output.shape
