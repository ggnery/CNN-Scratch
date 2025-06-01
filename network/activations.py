import torch
from .activation import Activation

class Tahn(Activation):
    def __init__(self, device: torch.device):
        def tanh(x: torch.Tensor):
            return torch.tanh(x)
        
        def tanh_prime(x: torch.Tensor):
            return 1 - torch.tanh(x) ** 2
        
        super().__init__(tanh, tanh_prime, device)
        
class Sigmoid(Activation):
    def __init__(self, device: torch.device):
        def sigmoid(x: torch.Tensor):
            return torch.sigmoid(x)
        
        def sigmoid_prime(x: torch.Tensor):
            s = sigmoid(x)
            return s * (1 - s)
        
        super().__init__(sigmoid, sigmoid_prime, device)      
