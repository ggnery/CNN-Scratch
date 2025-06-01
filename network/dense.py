from typing import Tuple
import torch
import math

from .layer import Layer

class Dense(Layer):
    weights: torch.Tensor
    bias: torch.Tensor
    
    def __init__(self, size: Tuple[int, int], device: torch.device):
        super().__init__(device)
        input_size = size[0]
        output_size = size[1]
        
        self.weights = torch.randn((output_size, input_size), device=device)/math.sqrt(input_size) # W ~ N(0, 1/sqtr(n_in))
        self.bias = torch.randn((output_size, 1), device=device)  # b ~ N(0, 1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.input = input
        return torch.matmul(self.weights, input) + self.bias # Y = W * X + B
    
    def backward(self, output_gradient: torch.Tensor, eta: float) -> torch.Tensor:
        weight_gradient = torch.matmul(output_gradient, torch.transpose(self.input, 0, 1)) # ∂E/∂W = ∂E/∂Y * X^T
        bias_gradient = output_gradient # ∂E/∂B = ∂E/∂Y 
        input_gradient = torch.matmul(torch.transpose(self.weights, 0, 1), output_gradient) # ∂E/∂X = W^T * ∂E/∂Y 
        
        #Stochastic Gradient Descend (SGD)
        self.weights -= eta * weight_gradient
        self.bias -= eta * bias_gradient
        
        return input_gradient