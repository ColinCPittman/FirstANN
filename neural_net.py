import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
       # Initialize weights with random values
       self.weights_input_hidden = np.random.randn(input_size, hidden_size)
       self.weights_hidden_output = np.random.randn(hidden_size, output_size)
       self.bias_hidden = np.random.randn(hidden_size)
       self.bias_output = np.random.randn(output_size)

    def forward(self, inputs):
        pass

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