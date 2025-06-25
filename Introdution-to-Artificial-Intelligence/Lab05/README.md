
# Neural Network Backpropagation: Sigmoid vs ReLU

## Project Overview

This project implements and compares two different activation functions (Sigmoid, ReLU) in a binary classification neural network, testing backpropagation on differently prepared datasets.

### Problem Statement

The network learns to classify 2D points based on whether both coordinates have the same sign:

- **Class 1**: Both coordinates positive OR both negative  
- **Class 0**: One positive, one negative coordinate

## Key Features

### Implementation

- Basic 2-layer neural network with configurable activation functions
- Multiple data preprocessing options (Raw, L1/L2 normalization)
- Comprehensive evaluation with confusion matrices and performance metrics
- **Improved weight initialization**:
  - Xavier/Glorot initialization for Sigmoid
  - He initialization for ReLU
- **Adaptive learning rate** with step decay schedule
- **Gradient clipping** to prevent exploding gradients
- **Numerical stability** improvements for sigmoid:
  - Clipping input to sigmoid to avoid overflow in `exp`

## Backpropagation Algorithm

The core of training this neural network relies on the **backpropagation algorithm**, which calculates gradients of the loss function with respect to weights and biases to iteratively minimize prediction error.

### Forward Pass

- Hidden Layer Input: `z1 = XW1 + b1`
- Hidden Layer Activation: `a1 = hidden_activation(z1)`
- Output Layer Input: `z2 = a1W2 + b2`
- Output Layer Activation: `a2 = hidden_activation(z2)`

### Backward Pass

- **Output Layer Error** (`delta2`):
  \[
  \delta^2 = \frac{2}{m}(a^2 - y_{\text{true}}) \cdot \text{sigmoid}'(z^2)
  \]
- **Output Gradients**:
  \[
  \frac{\partial \text{Loss}}{\partial W^2} = (a^1)^T \delta^2 \quad \quad
  \frac{\partial \text{Loss}}{\partial b^2} = \sum \delta^2
  \]
- **Hidden Layer Error** (`delta1`):
  \[
  \delta^1 = (\delta^2 (W^2)^T) \cdot \text{hidden\_activation}'(z^1)
  \]
- **Hidden Layer Gradients**:
  \[
  \frac{\partial \text{Loss}}{\partial W^1} = X^T \delta^1 \quad \quad
  \frac{\partial \text{Loss}}{\partial b^1} = \sum \delta^1
  \]

### Parameter Update (Gradient Descent)

- Weight update: `W_new = W_old - α * dLoss/dW`
- Bias update: `b_new = b_old - α * dLoss/db`

---

## Performance Results

### Results for learning rate 0.1

| **Model Configuration**     | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|-----------------------------|--------------|---------------|------------|--------------|
| Sigmoid + Raw Data          | 0.520        | 0.521         | 0.557      | 0.538        |
| ReLU + Raw Data             | 0.978        | 1.000         | 0.955      | 0.977        |
| Sigmoid + L1 Normalized     | 0.833        | 0.868         | 0.786      | 0.825        |
| ReLU + L1 Normalized        | 0.993        | 1.000         | 0.985      | 0.992        |
| Sigmoid + L2 Normalized     | 0.983        | 0.980         | 0.985      | 0.983        |
| ReLU + L2 Normalized        | 0.998        | 0.995         | 1.000      | 0.998        |

### Results for learning rate 0.2

| **Model Configuration**     | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|-----------------------------|--------------|---------------|------------|--------------|
| Sigmoid + Raw Data          | 0.855        | 0.841         | 0.896      | 0.868        |
| ReLU + Raw Data             | 0.970        | 0.972         | 0.972      | 0.972        |
| Sigmoid + L1 Normalized     | 0.910        | 0.855         | 1.000      | 0.922        |
| ReLU + L1 Normalized        | 0.985        | 0.972         | 1.000      | 0.986        |
| Sigmoid + L2 Normalized     | 0.990        | 1.000         | 0.981      | 0.995        |
| ReLU + L2 Normalized        | 0.995        | 1.000         | 0.991      | 0.995        |

### Results for learning rate 0.3

| **Model Configuration**     | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|-----------------------------|--------------|---------------|------------|--------------|
| Sigmoid + Raw Data          | 0.895        | 0.521         | 0.906      | 0.901        |
| ReLU + Raw Data             | 0.980        | 0.981         | 0.955      | 0.981        |
| Sigmoid + L1 Normalized     | 0.935        | 0.891         | 1.000      | 0.942        |
| ReLU + L1 Normalized        | 0.985        | 0.972         | 1.000      | 0.992        |
| Sigmoid + L2 Normalized     | 0.995        | 1.000         | 0.991      | 0.995        |
| ReLU + L2 Normalized        | 0.995        | 1.000         | 0.991      | 0.995        |


## Key Observations
### For learning rate 0.1
- **ReLU outperforms sigmoid**, especially with raw data, demonstrating better handling of non-linear patterns and faster convergence.
- **Sigmoid struggles with L1 normalization** (83% accuracy vs. ReLU's 99.3%), possibly due to gradient saturation or sensitivity to input scale.
- **L2 normalization performs better than L1 normalization** across both activation functions, showing better suitability for this network architecture.

### For learning rate 0.2 and 0.2
- **Both models are nearly identical on all data sets**

---
