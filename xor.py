from network import Network, Dense, Sigmoid, mse, mse_prime
import torch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X = torch.reshape(torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=device, dtype=torch.float32), (4, 2, 1))
    Y = torch.reshape(torch.tensor([[0], [1], [1], [0]], device=device, dtype=torch.float32), (4, 1, 1))
    X_test = torch.reshape(torch.tensor([[0, 0], [0.01, 0.99], [0.99, 0.01], [0.85, 0.75]], device=device, dtype=torch.float32), (4, 2, 1))

    layers = [
        Dense((2, 3), device),
        Sigmoid(device),
        Dense((3, 1), device),
        Sigmoid(device)
    ]
    
    network = Network(layers, mse, mse_prime, 0.1, 10000)    
    network.train(X, Y)
    
    for x_test in X_test: 
        result = network.eval(x_test)
        print(result)


if __name__ == "__main__":
    main()