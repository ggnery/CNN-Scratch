# CNN-Scratch 🧠

A **Convolutional Neural Network** implementation built from scratch using PyTorch tensors for computational efficiency while maintaining educational clarity.

## 📋 Overview

This project implements a complete CNN framework without relying on high-level neural network libraries. It demonstrates the fundamental concepts of deep learning by building layers, activation functions, loss functions, and training procedures from the ground up.

## ✨ Features

- **🔧 Custom Layer Architecture**: Implemented from scratch
  - Convolutional layers with forward/backward propagation
  - Dense (fully connected) layers
  - Activation layers (Sigmoid)
  - Reshape layers for transitioning between conv and dense layers

- **📊 Loss Functions**: 
  - Cross-entropy for classification
  - Mean Squared Error (MSE) for regression

- **🎯 Training Examples**:
  - MNIST digit classification
  - XOR problem solving

- **⚡ GPU Support**: CUDA acceleration when available

## 🏗️ Project Structure

```
CNN-Scratch/
├── network/
│   ├── __init__.py          # Package exports
│   ├── network.py           # Main Network class
│   ├── layer.py             # Base Layer class
│   ├── convolutional.py     # Convolutional layer implementation
│   ├── dense.py             # Dense layer implementation
│   ├── activation.py        # Activation layer wrapper
│   ├── activations.py       # Activation functions (sigmoid, etc.)
│   ├── losses.py            # Loss functions
│   └── reshape.py           # Reshape layer
├── mnist.py                 # MNIST classification example
├── xor.py                   # XOR problem example
└── data/                    # Dataset storage
```

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/CNN-Scratch.git
cd CNN-Scratch
```

2. **Install dependencies**:
```bash
pip install torch torchvision numpy scipy
```

## 📚 Usage

### MNIST Classification

Train a CNN on the MNIST dataset:

```bash
python mnist.py
```

The network architecture includes:
- Convolutional layer (1→5 filters, 3x3 kernels)
- Sigmoid activation
- Reshape layer (flatten for dense layers)
- Dense layer (3380→100 neurons)
- Dense output layer (100→10 classes)

### XOR Problem

Solve the classic XOR problem with a simple neural network:

```bash
python xor.py
```

## 🔬 Network Architecture

### Key Components

**Dense Layer**:
- **Forward pass**: 
  $$Y = W \cdot X + B$$

- **Backward pass**:
  1) $$\frac{\partial E}{\partial W} = \frac{\partial E}{\partial Y} \cdot X^T$$
  2) $$\frac{\partial E}{\partial X} = W^T \cdot \frac{\partial E}{\partial Y}$$
  3) $$\frac{\partial E}{\partial B} = \frac{\partial E}{\partial Y}$$

**Activation Layer**:
- **Forward pass**: 
  $$Y = f(X)$$

- **Backward pass**: 
  $$\frac{\partial E}{\partial X} = \frac{\partial E}{\partial Y} \odot f'(X)$$

**Convolutional Layer**:
- **Forward pass**: 
  $$Y_i = B_i + \sum_{j=0}^{n} X_j \star K_{ij}, \quad i = 0..d$$

- **Backward pass**:
  1) $$\frac{\partial E}{\partial K_{ij}} = X_j \star \frac{\partial E}{\partial Y_i}$$
  2) $$\frac{\partial E}{\partial B} = \frac{\partial E}{\partial Y}$$
  3) $$\frac{\partial E}{\partial X_j} = \sum_{i=0}^{n} \frac{\partial E}{\partial Y_i} \underset{\text{full}}{*} K_{ij}$$

**Training Loop**:
- Forward propagation through all layers
- Loss computation
- Backward propagation with gradient descent
- Parameter updates with learning rate

## 📐 Loss Functions & Activation Functions

### Loss Functions

**Mean Squared Error (MSE)**:
- **Forward**: 
  $$\text{MSE}(y_{true}, y_{pred}) = \frac{1}{n}\sum_{i=1}^{n}(y_{true} - y_{pred})^2$$

- **Derivative**: 
  $$\frac{\partial \text{MSE}}{\partial y_{pred}} = \frac{2(y_{pred} - y_{true})}{n}$$

**Cross Entropy**:
- **Forward**: 
  $$\text{CE}(y_{true}, y_{pred}) = -\sum_{i} [y_{true} \log(y_{pred}) + (1 - y_{true}) \log(1 - y_{pred})]$$

- **Derivative**: 
  $$\frac{\partial \text{CE}}{\partial y_{pred}} = \frac{1}{n}\left[\frac{1 - y_{true}}{1 - y_{pred}} - \frac{y_{true}}{y_{pred}}\right]$$

### Activation Functions

**Sigmoid**:
- **Forward**: 
  $$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- **Derivative**: 
  $$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Tanh**:
- **Forward**: 
  $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

- **Derivative**: 
  $$\tanh'(x) = 1 - \tanh^2(x)$$

## 📊 Example Results

### MNIST Performance
- Training epochs: 1 (configurable)
- Learning rate: 0.1
- Architecture: Conv → Sigmoid → Reshape → Dense → Sigmoid → Dense → Sigmoid

### XOR Performance
- Training epochs: 10,000
- Learning rate: 0.1
- Perfect convergence on XOR truth table

## 🛠️ Customization

You can easily modify the network by:

1. **Adding new layer types** in the `network/` directory
2. **Implementing new activation functions** in `activations.py`
3. **Creating custom loss functions** in `losses.py`
4. **Adjusting hyperparameters** in the example files

## 🎓 Educational Value

This implementation focuses on:
- Understanding CNN mathematics and operations
- Gradient computation and backpropagation
- Layer-by-layer neural network construction
- Transition from convolutional to dense layers
- GPU acceleration with PyTorch tensors

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new layer types
- Implement additional activation functions
- Improve documentation
- Add more examples

## 👨‍💻 Author

**Gabriel Nery** - [GitHub Profile](https://github.com/yourusername)

---

*Built with ❤️ for educational purposes and deep learning enthusiasts*

### 𝘖𝘮𝘯𝘦𝘴 𝘦𝘯𝘪𝘮 𝘊𝘩𝘳𝘪𝘴𝘵𝘶𝘴, 𝘯𝘪𝘩𝘪𝘭 𝘴𝘪𝘯𝘦 𝘔𝘢𝘳𝘪𝘢 