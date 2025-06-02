import torch
from .layer import Layer
from typing import Tuple
from scipy import signal

class Convolutional(Layer):
    input_shape: Tuple[int, int, int]
    output_shape: Tuple[int, int, int]
    kernels_shape: Tuple[int, int, int, int]
    
    input_depth: int
    depth: int
    
    kernels: torch.Tensor
    bias: torch.Tensor
    
    def __init__(self, input_shape: Tuple[int, int, int], kernel_size: int, depth: int, device: torch.device):
        super().__init__(device)
        input_depth, input_height, input_width = input_shape # Input is 3D: (LxHxW)
        self.depth = depth # #kernels = d
        self.input_shape = input_shape
        self.input_depth = input_depth
        
        # Y = I - K +1
        #The depth of the output is the same as the number of kernels
        #Think of output shape as 3 dimensional blocks
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1) # Output is 3D: (dxH'xW')

        self.kernels_shape = (depth,  input_depth, kernel_size, kernel_size) # Kernel is 4D: (dxLXkxk)
        
        self.kernels = torch.randn(self.kernels_shape, device=device) 
        self.bias = torch.randn(self.output_shape, device=device) # bias is 3D: (dxH'xW')
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.input = input
        self.output = self.bias.clone().to(self.device)
        
        #        n
        #Yi = Bi+∑ Xj ⋆ Kj, i = 0..d
        #        j
        for i in range(self.depth): # i = 0..d where d is the number of kernels   
            for j in range(self.input_depth): # j = 0..n where n is depth of the input
                # Xj ⋆ Kj
                self.output[i] += torch.tensor(signal.correlate2d(self.input[j].cpu().numpy(), self.kernels[i, j].cpu().numpy(), "valid"), device= self.device) 
        
        return self.output
    
    def backward(self, output_gradient: torch.Tensor, eta: float) -> torch.Tensor:
        kernels_gradient = torch.zeros(self.kernels_shape, device= self.device)
        input_gradient = torch.zeros(self.input_shape, device= self.device)
        
        for i in range(self.depth): # i = 0..d where d is the number of kernels   
            for j in range(self.input_depth): # j = 0..n where n is depth of the input 
                # ∂E/∂Kij = Xj ⋆ ∂E/∂Yi
                kernels_gradient[i, j] = torch.tensor(signal.correlate2d(self.input[j].cpu().numpy(), output_gradient[i].cpu().numpy(), "valid"), device=self.device) 
            
                #         n  
                #∂E/∂Xj = ∑ ∂E/∂Yi * Kij
                #         i      full 
                input_gradient[j] += torch.tensor(signal.convolve2d(output_gradient[i].cpu().numpy(), self.kernels[i, j].cpu().numpy(), "full"), device=self.device)
   
        self.kernels -= eta * kernels_gradient
        self.bias -= eta * output_gradient
        return input_gradient