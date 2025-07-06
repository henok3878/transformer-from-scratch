import torch
import pytest
from transformer.transformer import Transformer


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(0)


@pytest.mark.parametrize(
    "batch_size, src_seq_len, tgt_seq_len, d_model, d_ff, num_heads, src_vocab_size, tgt_vocab_size",
    [
        (2, 16, 12, 32, 64, 4, 100, 96),
        (3, 8, 8, 16, 32, 2, 50, 64),
    ],
)
def test_transformer_shape_and_dtype(
    batch_size,
    src_seq_len,
    tgt_seq_len,
    d_model,
    d_ff,
    num_heads,
    src_vocab_size,
    tgt_vocab_size,
):
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.0,
    )
    src_ids = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt_ids = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    # no masks
    out = model(
        src_ids,
        tgt_ids,
        src_mask=None,
        tgt_mask=None,
        kv_mask=None,
    )

    assert out.shape == (batch_size, tgt_seq_len, tgt_vocab_size)
    assert out.dtype == torch.float32


def test_transformer_mask_optional_and_broadcasting(
    batch_size=2,
    src_seq_len=12,
    tgt_seq_len=9,
    d_model=16,
    d_ff=32,
    num_heads=4,
    src_vocab_size=20,
    tgt_vocab_size=20,
):
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0.0,
    )
    src_ids = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt_ids = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    # 2D masks
    src_mask = torch.ones((src_seq_len, src_seq_len), dtype=torch.bool)
    tgt_mask = torch.ones((tgt_seq_len, tgt_seq_len), dtype=torch.bool)
    kv_mask = torch.ones((tgt_seq_len, src_seq_len), dtype=torch.bool)

    # should run without error
    _ = model(
        src_ids,
        tgt_ids,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
        kv_mask=kv_mask,
    )


def test_transformer_backprop_smoke(
    batch_size=2,
    src_seq_len=8,
    tgt_seq_len=5,
    d_model=16,
    d_ff=32,
    num_heads=4,
    src_vocab_size=15,
    tgt_vocab_size=20,
):
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
    )
    src_ids = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt_ids = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    out = model(src_ids, tgt_ids, None, None, None)
    loss = out.sum()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert any((g.abs().sum() > 0) for g in grads)
