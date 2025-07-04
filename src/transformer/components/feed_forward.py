import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        # linear1 shape: (d_model, d_ff)
        # output1 shape: (batch_size,seq_len, d_ff)
        output1 = self.relu(self.linear1(x))

        # output2 shape: (batch_size, seq_len, d_model)
        output2 = self.linear2(self.dropout(output1))

        return output2
