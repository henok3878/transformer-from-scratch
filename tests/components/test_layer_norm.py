import torch
from transformer.components.layer_norm import LayerNorm


def test_forward_shape():
    batch_size, seq_len, d_model = 4, 16, 6
    x = torch.randn(batch_size, seq_len, d_model)
    ln = LayerNorm(d_model)

    assert ln(x).size() == x.size()
