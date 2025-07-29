import torch 
import torch.nn as nn 
from abc import ABC, abstractmethod  

class BaseAttention(nn.Module, ABC):
    
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