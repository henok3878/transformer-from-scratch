import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape: (batch_size, seq_len, d_model)

        Returns:
            Normalized `x` over the last dim to zero mean and unit variance with
            scale and shfit applied by learnable weight and bias.
        """
        mean = x.mean(dim=x.size(-1), keepdim=True)  # same shape with x
        std = x.std(dim=x.size(-1), keepdim=True)  # same shape with x

        return self.weight * (x - mean) / (std + self.eps) + self.bias
