# DL-Assignment-1
# README: Flexible Feedforward Neural Network for EMNIST Digits

## Overview

This project implements a flexible feedforward neural network to classify images from the **EMNIST Digits dataset**. The model supports different configurations of **hidden layers, activation functions, optimizers, learning rates, weight initializations, and loss functions**. It compares **Cross-Entropy Loss** and **Mean Squared Error Loss** to analyze their effectiveness.

## Features

- **Dataset Handling:** Loads EMNIST Digits dataset and splits it into train (54,000), validation (6,000), and test (10,000) sets.
- **Customizable Network:** Allows changing **number of hidden layers, neurons per layer, activation functions, and weight initialization methods**.
- **Backpropagation & Optimization:** Supports multiple optimizers:
  - SGD
  - Momentum-based Gradient Descent
  - Nesterov Accelerated Gradient Descent
  - RMSprop
  - Adam
- **Evaluation Metrics:** Computes **validation accuracy, test accuracy, and confusion matrix**.
- **Comparison of Loss Functions:** Analyzes differences between **Cross-Entropy Loss** and **Mean Squared Error Loss**.

## Dataset

The dataset is automatically downloaded using `torchvision.datasets.EMNIST`.

## Configuration Hyperparameters

The model was trained using various hyperparameters:

- **Hidden Layers:** 3, 4, 5
- **Neurons per Layer:** 32, 64, 128
- **Weight Decay (L2 Regularization):** 0, 0.0005, 0.5
- **Learning Rate:** 1e-3, 1e-4
- **Optimizers:** SGD, Momentum, Nesterov, RMSprop, Adam, Nadam
- **Batch Sizes:** 16, 32, 64
- **Activation Functions:** ReLU, Sigmoid
- **Weight Initializations:** Random, Xavier

## Results & Findings

### Configurations

| Configuration | Hidden Layers      | Optimizer | Batch Size | Activation | Val Accuracy | Test Accuracy |
| ------------- | ------------------ | --------- | ---------- | ---------- | ------------ | ------------- |
| Config 1      | [32, 64, 128]      | Adam      | 32         | ReLU       | 96.25%       | 96.17%        |
| Config 2      | [32, 64, 128, 256] | Momentum  | 64         | ReLU       | 93.25%       | 93.78%        |
| Config 3      | [64, 128, 256]     | RMSprop   | 16         | Sigmoid    | 11.05%       | 11.35%        |

## Conclusion & Recommendations for MNIST

Based on the results, the following recommendations for the **MNIST dataset**:

1. **Use Adam Optimizer with ReLU Activation:** Adam consistently performed well, achieving over **96% accuracy**.
2. **Xavier Initialization with Moderate Hidden Layers (3-4):** Too many layers increase complexity without significant accuracy gains. Xavier initialization stabilizes training.
3. **Batch Size of 32-64 with L2 Regularization (0.0005):** This balance provides stable training and generalization without overfitting.
