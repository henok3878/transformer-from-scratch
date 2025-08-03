import torch
import torch.nn as nn
from typing import Type 
from transformer.components.base.attention import BaseAttention
from transformer.components.base.feed_forward import BaseFeedForward
from transformer.components.encoder import Encoder
from transformer.components.decoder import Decoder
from transformer.components.feed_forward import PositionwiseFeedForward
from transformer.components.input_embedding import InputEmbedding
from transformer.components.layer_norm import LayerNorm
from transformer.components.multi_head import MultiHeadAttention
from transformer.components.positional_encoding import PositionalEncoding
from transformer.config import AppConfig, DataConfig, ModelConfig, TokenizationStrategy

class Transformer(nn.Module):
    def __init__(
        self, 
        model_config: ModelConfig,
        data_config: DataConfig, 
        attention_cls: Type[BaseAttention] | None = None,
        feedforward_cls: Type[BaseFeedForward] | None = None,
        norm_cls: Type[nn.Module] | None = None,
        positional_encoding_cls: Type[nn.Module] | None = None,
        use_input_positional_encoding: bool = True,
        **kwargs
        ):
        super().__init__()

        if attention_cls is None:
            attention_cls = MultiHeadAttention
        if feedforward_cls is None:
            feedforward_cls = PositionwiseFeedForward
        if norm_cls is None:
            norm_cls = LayerNorm
        if use_input_positional_encoding and positional_encoding_cls is None:
            positional_encoding_cls = PositionalEncoding

        is_joint_vocab = data_config.tokenization_strategy == TokenizationStrategy.JOINT
        
        if use_input_positional_encoding:
            assert positional_encoding_cls is not None
            self.src_embedding = nn.Sequential(
                InputEmbedding(model_config.src_vocab_size, model_config.d_model),
                positional_encoding_cls(
                    d_model=model_config.d_model,
                    seq_len=model_config.src_max_len,
                    dropout=model_config.dropout,
                ),
            )
        else:
            self.src_embedding = InputEmbedding(model_config.src_vocab_size, model_config.d_model)

        if is_joint_vocab:
            self.tgt_embedding = self.src_embedding
        else:
            if use_input_positional_encoding:
                assert positional_encoding_cls is not None
                self.tgt_embedding = nn.Sequential(
                    InputEmbedding(model_config.tgt_vocab_size, model_config.d_model),
                    positional_encoding_cls(
                        d_model=model_config.d_model,
                        seq_len=model_config.tgt_max_len,
                        dropout=model_config.dropout,
                    ),
                )
            else:
                self.tgt_embedding = InputEmbedding(model_config.tgt_vocab_size, model_config.d_model)

        self.encoder = Encoder(
            num_layers=model_config.num_encoder_layers,
            d_model=model_config.d_model,
            d_ff=model_config.d_ff,
            num_heads=model_config.num_heads,
            dropout=model_config.dropout,
            attention_cls=attention_cls,
            feedforward_cls=feedforward_cls,
            norm_cls=norm_cls,
            **kwargs
        )

        self.decoder = Decoder(
            num_layers=model_config.num_decoder_layers,
            d_model=model_config.d_model,
            d_ff=model_config.d_ff,
            num_heads=model_config.num_heads,
            dropout=model_config.dropout,
            attention_cls=attention_cls,
            feedforward_cls=feedforward_cls,
            norm_cls=norm_cls,
            **kwargs
        )

        self.output_proj = nn.Linear(model_config.d_model, model_config.tgt_vocab_size)

        self._init_weights()

        self.output_proj.weight = self.tgt_embedding[0].embedding.weight # type: ignore  

    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod 
    def load_from_checkpoint(checkpoint_path: str, config: AppConfig, device: torch.device = torch.device("cpu")):
        model = Transformer(config.model, config.data).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # remove '_orig_mod.' prefix if present
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.eval()
        return model

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
