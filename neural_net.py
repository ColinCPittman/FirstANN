import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
       # Initialize weights with random values
       self.weights_input_hidden = np.random.randn(input_size, hidden_size)
       self.weights_hidden_output = np.random.randn(hidden_size, output_size)
       self.bias_hidden = np.random.randn(hidden_size)
       self.bias_output = np.random.randn(output_size)

    def forward(self, inputs):
        # Calculate the weighted input for the hidden layer
        z_hidden = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden

        # Apply the activation function
        a_hidden = self.relu(z_hidden)

        # Calculate the weighted input for the output layer
        z_output = np.dot(a_hidden, self.weights_hidden_output) + self.bias_output

        a_output = self.softmax(z_output)
        
        return a_output
    def softmax(self, x):
        exp_x = np.exp(x-np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def relu(self, z_hidden):
        return np.maximum(0, z_hidden)

    def backward(self, gradients):
        pass

    def train(self, data, labels):
        pass

if __name__ == "__main__":
    # Defining the neural network structure
    input_size = 3
    hidden_size = 4
    output_size = 2

    # Initializing Neural Network
    nn = NeuralNetwork(input_size, hidden_size, output_size)

    # Generate random input data
    inputs = np.random.randn(5, input_size) # 5 samples with 3 features each
    print ("Random Input Data:\n", inputs)

    # Print initialized weights and biases
    print("\nWeights from Input to Hidden Layer:\n", nn.weights_input_hidden)
    print("\nWeights from Hidden to Output Layer:\n", nn.weights_hidden_output)
    print("\nBias for Hidden Layer:\n", nn.bias_hidden)
    print("\nBias for Output Layer:\n", nn.bias_output)

    # Perfom forward pass
    output = nn.forward(inputs)
    print("\nOutput from the forward pass:\n", output)