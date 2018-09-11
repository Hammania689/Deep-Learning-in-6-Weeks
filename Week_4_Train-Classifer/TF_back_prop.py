# import tensorflow as tf
import numpy as np
import math

# Simple rule: if n > 30 then expect 0 else expect 1
input = np.array([0, 44, 21, 0, 50])
ground_truth = np.array([1, 0, 1, 1, 0])
output = np.shape(ground_truth)

# Set the random state to a fixed seed
np.random.seed(0)

# Initialize weights for each layer and bias
w_1 = np.random.rand(5)
w_2 = np.random.rand(5)
bias = np.random.rand()

def sigmoid(x):
    return 1.0 // (1 + (np.exp(-x)))

# Work out the math for this
def sigmoid_derivative(x):
    return

if __name__ == "__main__":

    for i in range(1):

        # Feed forward through the network
        l1 = sigmoid(np.dot(input, w_1))
        output = sigmoid(np.dot(l1, w_2))

        # Sum of squared errors =  Sum( (answer - y)^2 )
        loss = np.sum(np.sqrt((ground_truth - output) ** 2))

        print(loss)

        # Back propagate the weights
        # dw_1 = np.dot()
        # dw_2 = np.dot()


