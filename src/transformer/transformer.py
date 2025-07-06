import torch
import torch.nn as nn
from transformer.components.encoder import Encoder
from transformer.components.decoder import Decoder
from transformer.components.input_embedding import InputEmbedding
from transformer.components.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout: float = 0.1,
        src_max_len: int = 5000,
        tgt_max_len: int = 5000,
    ):
        super().__init__()
        self.src_embedding = nn.Sequential(
            InputEmbedding(src_vocab_size, d_model),
            PositionalEncoding(d_model=d_model, seq_len=src_max_len, dropout=dropout),
        )
        self.tgt_embedding = nn.Sequential(
            InputEmbedding(tgt_vocab_size, d_model),
            PositionalEncoding(d_model=d_model, seq_len=tgt_max_len, dropout=dropout),
        )

        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            src_ids: source token indices of shape:
            tgt_ids: target token indices of shape:
            src_mask: (src_seq_len, src_seq_len) or (batch_size, 1, src_seq_len, src_seq_len) or (batch_size, num_heads, src_seq_len, src_seq_len)
            tgt_mask: (tgt_seq_len, tgt_seq_len) or (batch_size, 1, tgt_seq_len, tgt_seq_len) or (batch_size, num_heads, tgt_seq_len, tgt_seq_len)
            kv_mask: (tgt_seq_len, src_seq_len) or (batch_size, 1, tgt_seq_len, src_seq_len) or (batch_size, num_heads, tgt_seq_len, src_seq_len)

        Returns:
            logits of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # embed & encode
        src_emb = self.src_embedding(src_ids)  # (batch_size, src_seq_len, d_model)
        kv = self.encoder(src_emb, src_mask)

        # embed & decode
        tgt_emb = self.tgt_embedding(tgt_ids)  # (batch_size, tgt_seq_len, d_model)
        dec_out = self.decoder(tgt_emb, kv, target_mask=tgt_mask, kv_mask=kv_mask)

        # project to vocabulary
        logits = self.output_proj(dec_out)  # (batch_size, tgt_seq_len, vocab_size)
        return logits
