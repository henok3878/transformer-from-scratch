import torch
import torch.nn as nn
from transformer.components.encoder import Encoder
from transformer.components.decoder import Decoder
from transformer.components.input_embedding import InputEmbedding
from transformer.components.positional_encoding import PositionalEncoding
from config import DataConfig, ModelConfig, TokenizationStrategy

class Transformer(nn.Module):
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()

        is_joint_vocab = data_config.tokenization_strategy == TokenizationStrategy.JOINT
         
        self.src_embedding = nn.Sequential(
            InputEmbedding(model_config.src_vocab_size, model_config.d_model),
            PositionalEncoding(
                d_model=model_config.d_model,
                seq_len=model_config.src_max_len,
                dropout=model_config.dropout,
            ),
        )
        if is_joint_vocab:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = nn.Sequential(
                InputEmbedding(model_config.tgt_vocab_size, model_config.d_model),
                PositionalEncoding(
                    d_model=model_config.d_model,
                    seq_len=model_config.tgt_max_len,
                    dropout=model_config.dropout,
                ),
            )

        self.encoder = Encoder(
            num_layers=model_config.num_encoder_layers,
            d_model=model_config.d_model,
            d_ff=model_config.d_ff,
            num_heads=model_config.num_heads,
            dropout=model_config.dropout,
        )

        self.decoder = Decoder(
            num_layers=model_config.num_decoder_layers,
            d_model=model_config.d_model,
            d_ff=model_config.d_ff,
            num_heads=model_config.num_heads,
            dropout=model_config.dropout,
        )

        self.output_proj = nn.Linear(model_config.d_model, model_config.tgt_vocab_size)

        self.output_proj.weight = self.tgt_embedding[0].embedding.weight # type: ignore  

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
