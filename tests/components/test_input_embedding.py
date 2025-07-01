import torch
from src.components.input_embedding import InputEmbedding


def test_forward_shape():
    vocab_size, d_model = 65, 16
    ie = InputEmbedding(vocab_size, d_model)

    batch_size, seq_len = 4, 8
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    out = ie(x)
    assert out.shape == (batch_size, seq_len, d_model)
