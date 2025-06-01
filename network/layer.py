import torch
from abc import ABC, abstractmethod

class Layer(ABC):
    device: torch.device
    input: torch.Tensor
    output: torch.Tensor
    
    def __init__(self, device: torch.device):
        self.input = None
        self.output = None
        self.device = device
    
    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def backward(self, output_gradient: torch.Tensor, eta: float) -> torch.Tensor:
        pass
        
    
    
    