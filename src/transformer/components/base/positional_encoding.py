import torch
import torch.nn as nn 
from abc import ABC, abstractmethod

class BasePositionalEncoding(nn.Module, ABC):
    """
    Base class for all positional encoding modules.

    All positional encoding modules should implement:
        - __init__(self, seq_len: int, d_model: int, dropout: float = 0.1, **kwargs)
        - forward(self, x: torch.Tensor) -> torch.Tensor
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass 