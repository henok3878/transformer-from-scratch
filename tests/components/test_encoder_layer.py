import torch
from transformer.components.encoder_layer import EncoderLayer
from transformer.components.feed_forward import PositionwiseFeedForward
from transformer.components.layer_norm import LayerNorm
from transformer.components.multi_head import MultiHeadAttention


def test_forward_shape():
    batch_size, seq_len, d_model, d_ff, num_heads = 4, 32, 16, 32, 4
    encoder_block = EncoderLayer(
        d_model,
        d_ff,
        num_heads,
        attention_cls=MultiHeadAttention, feedforward_cls=PositionwiseFeedForward, norm_cls=LayerNorm
    )

    x = torch.randn(batch_size, seq_len, d_model)
    encoder_output = encoder_block(x)

    assert x.shape == encoder_output.shape
