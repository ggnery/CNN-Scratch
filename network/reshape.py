from .layer import Layer
import torch

class Reshape(Layer):
    input_shape: torch.Tensor
    output_shape: torch.Tensor
    
    def __init__(self, input_shape: torch.Tensor, output_shape: torch.Tensor, device: torch.device):
        super().__init__(device)
        
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.reshape(input, self.output_shape)
    
    def backward(self, output_gradient: torch.Tensor, eta: float) -> torch.Tensor:
        return torch.reshape(output_gradient, self.input_shape)