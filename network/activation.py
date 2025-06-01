import torch
from typing import Callable
from .layer import Layer

class Activation(Layer):
    f: Callable[[torch.Tensor], torch.Tensor]
    f_prime: Callable[[torch.Tensor], torch.Tensor]
    
    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor], 
                 f_prime: Callable[[torch.Tensor], torch.Tensor], 
                 device: torch.device):
        super().__init__(device)
        
        self.f = f
        self.f_prime = f_prime
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.input = input
        return self.f(input) # Y = f(X)
    
    def backward(self, output_gradient: torch.Tensor, eta: float) -> torch.Tensor:
        return output_gradient * self.f_prime(self.input) # ∂E/∂X = ∂E/∂Y ⊙ f'(X)