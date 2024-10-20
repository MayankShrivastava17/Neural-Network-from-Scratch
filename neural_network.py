import random
import math

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights and biases
def initialize_parameters(input_size, hidden_size, output_size):
    weights_input_hidden = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
    weights_hidden_output = [random.uniform(-1, 1) for _ in range(hidden_size)]
    bias_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]
    bias_output = random.uniform(-1, 1)
    
    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

def forward_pass(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    # Hidden layer activations
    hidden_layer_activations = []
    for i in range(len(weights_input_hidden[0])):
        activation = sum(inputs[j] * weights_input_hidden[j][i] for j in range(len(inputs))) + bias_hidden[i]
        hidden_layer_activations.append(sigmoid(activation))
    
    # Output layer activation
    output_activation = sum(hidden_layer_activations[i] * weights_hidden_output[i] for i in range(len(hidden_layer_activations))) + bias_output
    output = sigmoid(output_activation)
    
    return hidden_layer_activations, output

def backward_pass(inputs, hidden_layer_activations, output, target, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, learning_rate):
    # Calculate output error
    output_error = target - output
    output_delta = output_error * sigmoid_derivative(output)
    
    # Update weights and bias for output layer
    for i in range(len(weights_hidden_output)):
        weights_hidden_output[i] += hidden_layer_activations[i] * output_delta * learning_rate
    
    bias_output += output_delta * learning_rate

    # Calculate hidden layer error
    hidden_errors = [output_delta * weights_hidden_output[i] for i in range(len(weights_hidden_output))]
    
    # Update weights and bias for hidden layer
    for i in range(len(weights_input_hidden)):
        for j in range(len(weights_input_hidden[0])):
            weights_input_hidden[i][j] += inputs[i] * hidden_errors[j] * sigmoid_derivative(hidden_layer_activations[j]) * learning_rate
        bias_hidden[j] += hidden_errors[j] * learning_rate

def train(inputs, targets, epochs, learning_rate):
    weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = initialize_parameters(2, 2, 1)
    
    for epoch in range(epochs):
        for input_vector, target in zip(inputs, targets):
            hidden_layer_activations, output = forward_pass(input_vector, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
            backward_pass(input_vector, hidden_layer_activations, output, target, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, learning_rate)
    
    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

def predict(input_vector, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_layer_activations, output = forward_pass(input_vector, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
    return output

# Sample dataset (XOR problem)
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets = [0, 1, 1, 0]  # XOR output

# Train the neural network
epochs = 10000
learning_rate = 0.1
weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = train(inputs, targets, epochs, learning_rate)

# Make predictions
for input_vector in inputs:
    prediction = predict(input_vector, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
    print(f'Input: {input_vector}, Prediction: {prediction}')
