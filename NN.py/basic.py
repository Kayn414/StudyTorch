import numpy as np

# Step 1: Define the activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # Sigmoid function: squashes values to be between 0 and 1

def sigmoid_derivative(x):
    return x * (1 - x)  # Derivative of sigmoid: used for updating weights

# Step 2: Create the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Randomly initialize weights for the layers
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        
        # Initialize biases (optional, but helpful)
        self.bias_hidden = np.random.rand(1, hidden_size)
        self.bias_output = np.random.rand(1, output_size)

    def feedforward(self, X):
        # Step 3: Feedforward phase
        # Calculate hidden layer output
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        
        # Calculate output layer output
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        output = sigmoid(self.output_layer_input)
        return output

    def backpropagation(self, X, y, output):
        # Step 4: Backpropagation phase
        # Calculate error
        output_error = y - output  # Difference between desired output and the prediction
        output_delta = output_error * sigmoid_derivative(output)  # Gradient for output layer
        
        # Calculate hidden layer error
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer_output)  # Gradient for hidden layer

        # Update weights and biases
        self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_delta)
        self.weights_input_hidden += np.dot(X.T, hidden_delta)
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True)
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs):
        # Step 5: Training loop
        for _ in range(epochs):
            output = self.feedforward(X)  # Forward pass
            self.backpropagation(X, y, output)  # Backward pass

# Step 6: Prepare input data
# Input (X): 4 samples, each with 3 features
# Output (y): 4 samples, each with 1 output (0 or 1)
X = np.array([[0, 0, 1],
              [1, 1, 1],
              [1, 0, 1],
              [0, 1, 1]])

y = np.array([[0], [1], [1], [0]])

# Step 7: Initialize and train the neural network
nn = NeuralNetwork(input_size=3, hidden_size=4, output_size=1)
nn.train(X, y, epochs=10000)

# Step 8: Test the trained network
print("Predicted output after training:")
print(nn.feedforward(X))
