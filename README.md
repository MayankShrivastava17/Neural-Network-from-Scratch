# Neural Network Implementation for Scratch 

This repository contains a simple implementation of a feedforward neural network that solves the XOR problem. The network utilizes a single hidden layer and employs the sigmoid activation function.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
  - [Activation Functions](#activation-functions)
  - [Initialization](#initialization)
  - [Forward Pass](#forward-pass)
  - [Backward Pass](#backward-pass)
  - [Training](#training)
  - [Prediction](#prediction)
- [Sample Dataset](#sample-dataset)
- [License](#license)

## Overview

The XOR problem is a classic problem in machine learning, where a neural network must learn the exclusive OR function. The inputs are pairs of binary values, and the output is also a binary value representing the XOR of the inputs.

## Installation

To run this code, you need Python installed on your machine. This implementation does not require any external libraries beyond the standard library.

## Usage

Simply run the Python script. It will train the neural network on the XOR dataset for a specified number of epochs and display the predictions for each input.

```bash
python neural_network.py
```

## Code Explanation

### Activation Functions

- **Sigmoid Function**: The sigmoid function maps any real-valued number into the range (0, 1). It's used as the activation function in both the hidden and output layers.
  
  ```python
  def sigmoid(x):
      return 1 / (1 + math.exp(-x))
  ```

- **Derivative of Sigmoid**: This function is used to calculate the gradient during backpropagation.
  
  ```python
  def sigmoid_derivative(x):
      return x * (1 - x)
  ```

### Initialization

This function initializes the weights and biases randomly for the input-to-hidden layer and the hidden-to-output layer.

```python
def initialize_parameters(input_size, hidden_size, output_size):
    ...
```

### Forward Pass

The forward pass computes the output of the network given an input vector. It calculates the activations of the hidden layer and the final output.

```python
def forward_pass(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    ...
```

### Backward Pass

This function implements the backpropagation algorithm. It calculates the error and updates the weights and biases based on the gradient of the loss function.

```python
def backward_pass(inputs, hidden_layer_activations, output, target, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, learning_rate):
    ...
```

### Training

The `train` function orchestrates the training process by iterating over multiple epochs and updating weights and biases.

```python
def train(inputs, targets, epochs, learning_rate):
    ...
```

### Prediction

This function makes predictions based on the trained weights and biases. It utilizes the forward pass to compute the output for a given input vector.

```python
def predict(input_vector, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    ...
```

## Sample Dataset

The following sample dataset represents the XOR problem:

```python
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets = [0, 1, 1, 0]  # XOR output
```

## License

This project is licensed under the MIT License. Feel free to modify and distribute the code as needed.
