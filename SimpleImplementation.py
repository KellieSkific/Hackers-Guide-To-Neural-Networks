import numpy as np

# Defining the sigmoid function - that takes an input of x
def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

# Defining the derivative sigmoid function - that takes an input of x
def sigmoid_derivative(x):
    return x * (1.0 - x)

# Creating A Neural Network Class
class NeuralNetwork:
    def __init__(self, x, y):
        # Initializing the Weights and assuming the Biases to be zero
        self.input = x
        self.weight1 = np.random.rand(self.input.shape[1],4)
        self.weight2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(self.y.shape)

# ForwardPropagation Algorithm
    def forwardPropagation(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weight1))
        self.output = sigmoid(np.dot(self.layer1, self.weight2))

# BackPropagation Algorithm
    def backPropagation(self):
        # Application of the Chain Rule to find derivative of the loss function
        # with respect to weight 1 and weight 2
        derivative_weight2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        derivative_weight1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weight2.T) * sigmoid_derivative(self.layer1)))

        # Updating the weights with the derivative (slope) of the loss function
        self.weight1 += derivative_weight1
        self.weight2 += derivative_weight2


if __name__ == "__main__":
    x = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])

    nn = NeuralNetwork(x,y)

# Training the neural network for 3000 iterations
    for i in range(3000):
        nn.forwardPropagation()
        nn.backPropagation()

    print(nn.output)