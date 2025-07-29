import torch
import torch.nn as nn 
from abc import ABC, abstractmethod

class BasePositionalEncoding(nn.Module, ABC):

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass 