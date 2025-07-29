import torch 
import torch.nn as nn 
from abc import ABC, abstractmethod

class BaseFeedForward(nn.Module, ABC):
    """
    Base class for all feedforward modules.

    All feedforward modules should implement:
        - __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, **kwargs)
        - forward(self, x: torch.Tensor) -> torch.Tensor
    """
    @abstractmethod 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass 