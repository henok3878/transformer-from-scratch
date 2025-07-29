import math
import torch
import torch.nn as nn

from transformer.components.base.positional_encoding import BasePositionalEncoding


class PositionalEncoding(BasePositionalEncoding):
    def __init__(self, seq_len: int, d_model: int, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pos_encoding = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, 1, dtype=torch.float).unsqueeze(
            1
        )  # shape: (seq_len, 1)
        evens_i = torch.arange(0, d_model, 2)
        # div_terms[i] = 1 / (10000 ** (2i / d_model)), which is equivalent to
        # exp(-ln(10000) * (2i/ d)) -> exp((-ln(10000)/d) *2i)
        div_term = torch.exp(-math.log(10000.0) / d_model * evens_i)  # (d_model,)
        # pos * div_term has shape (seq_len, d_model/2)
        pos_encoding[:, 0::2] = torch.sin(pos * div_term)
        pos_encoding[:, 1::2] = torch.cos(pos * div_term)
        # add batchsize: (seq_len, d_model) -> (1, seq_len, d_model)
        pos_encoding = pos_encoding.unsqueeze(0)

        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  the output of the input embedding layer.
                shape: (batch_size, seq_len, d_model)

        Returns:
            the sums of the embeddings and positional encodings with dropout applied
            shape: (batch_size, seq_len, d_model).
        """
        curr_seq_len = x.size(1)
        x = x + self.pos_encoding[:, :curr_seq_len, :]  # (batch_size, seq_len, d_model)
        return self.dropout(x)
