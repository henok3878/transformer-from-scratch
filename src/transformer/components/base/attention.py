import torch 
import torch.nn as nn 
from abc import ABC, abstractmethod  

class BaseAttention(nn.Module, ABC):
    """
    Base class for all attention modules. 
    
    All attention modules should implement:
        - __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, **kwargs)
        - forward(self, x: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor
    """
    
    @abstractmethod
    def forward(
        self, x: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: shape = (batch_size, x_seq_len, d_model)
            kv: shape = (batch_size, kv_seq_len, d_model)
            mask: shape = (batch_size, 1, x_seq_len, kv_seq_len)
        """
        pass 