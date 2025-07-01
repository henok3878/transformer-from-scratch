import torch
from src.components.positional_encoding import PositionalEncoding


def test_forward_shape():
    batch_size, seq_len, d_model = 4, 8, 16

    pe = PositionalEncoding(seq_len, d_model)

    x = torch.randn(batch_size, seq_len, d_model)
    out = pe(x)
    assert out.shape == x.shape
