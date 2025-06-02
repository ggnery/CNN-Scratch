import torch
from torchvision import datasets, transforms
from network import Dense, Sigmoid, Convolutional, Reshape, Network, cross_entropy, cross_entropy_prime

# load MNIST using PyTorch
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

def preprocess_data(x: torch.Tensor, y: torch.Tensor, device: torch.device):
        x = x.reshape(x.shape[0], 1, 28, 28)
        x = x.to(torch.float32) / 255
        y = torch.nn.functional.one_hot(y, num_classes=10)
        y = y.reshape(y.shape[0], 10, 1)
        
        x.to(device)
        y.to(device)
        
        return x, y
    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_train, y_train = preprocess_data(train_dataset.data, train_dataset.targets, device)
    x_test, y_test = preprocess_data(test_dataset.data, test_dataset.targets, device)
    
    layers = [
        Convolutional((1, 28, 28), 3, 5, device),
        Sigmoid(device),
        
        Convolutional((5, 26, 26), 3, 5, device),
        Sigmoid(device),
        
        Convolutional((5, 24, 24), 3, 5, device),
        Sigmoid(device),
        
        Reshape((5, 22, 22), (5 * 22 * 22, 1), device),
        
        Dense((5 * 22 * 22, 100), device),
        Sigmoid(device),
        
        Dense((100, 10), device),
        Sigmoid(device)    
    ]
    
    network = Network(layers, cross_entropy, cross_entropy_prime, 0.1, 1)
    network.train(x_train, y_train)
    
    accuracy = network.accuracy(x_test, y_test)
    loss = network.mean_cost(x_test, y_test)
    
    print(f"Final cost on test data: {loss}")
    print(f"Final Acurracy test data: {accuracy}")
    

if __name__ == "__main__":
    main()