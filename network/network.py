from typing import List, Callable
from .layer import Layer
import torch

class Network:
    layers: List[Layer]
    loss: Callable[[torch.Tensor, torch.Tensor], float]
    loss_prime: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    eta: float
    epochs: int
    
    def __init__(self, layers: List[Layer], 
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
                 loss_prime: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 eta: float,
                 epochs: int):
        self.layers = layers
        self.loss = loss
        self.loss_prime = loss_prime
        self.eta = eta
        self.epochs = epochs
    
    def train(self, x_train, y_train, verbose: bool = True):
        for epoch in range(self.epochs):
            loss_sum = 0
            for x, y in zip(x_train, y_train):
                y_pred = self.eval(x) # forward
                loss_sum += self.loss(y, y_pred) # loss 
                
                # backward
                output_gradient = self.loss_prime(y, y_pred) # ∂E/∂Y in last layer
                for layer in reversed(self.layers):
                    output_gradient = layer.backward(output_gradient, self.eta) 
            
            if verbose:
                print(f"Loss in epoch {epoch + 1}: {loss_sum/len(x_train)}")
            
 
    def eval(self, input: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            input = layer.forward(input)
            
        return input