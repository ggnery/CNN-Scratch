import torch

def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torch.mean((y_true - y_pred)**2)

def mse_prime(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return 2*(y_pred - y_true) / y_true.size(0)

def cross_entropy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torch.sum(-y_true * torch.log(y_pred) - (1 - y_true) * torch.log(1 - y_pred))

def cross_entropy_prime(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return ( (1 - y_true)/(1 - y_pred) - (y_true/y_pred) ) / y_true.size(0)