import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the embedding vectors for x.

        Maps input token IDs to their corresponding embedding vectors.

        Args:
            x:  tensor of tokens of shape (batch_size, seq_len)
                batch_size: determines the number of input sequence we are processing at the same time.
                seq_len: refers to the sequence length of the input in the current batch.

        Returns:
            The embedding vectors of shape: (batch_size, seq_len, d_model)
        """
        return self.embedding(x) * math.sqrt(self.d_model)
