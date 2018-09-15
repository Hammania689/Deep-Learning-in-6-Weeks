# import tensorflow as tf
import numpy as np
import math

# Simple rule: (n * n) + 1
input = np.array([1, 2, 3, 4, 5])
ground_truth = np.array([0, 0, 1, 1, 1]).T
output = np.shape(ground_truth)

# Set the random state to a fixed seed
np.random.seed(0)

# Initialize weights for each layer and bias
w_1 = 2 * np.random.rand(5) - 1
w_2 = 2 * np.random.rand(5).T - 1
bias = np.random.rand()

def sigmoid(x):
    return 1 / (1 + (np.exp(-x)))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

if __name__ == "__main__":

    for i in range(60000):

        """
            @TODO work out math and find a nicer way to implement gradient descent
        """
        # Feed forward through the network
        l1 = sigmoid(np.dot(input, w_1))
        output = sigmoid(np.dot(l1, w_2))

        # Sum of squared errors =  Sum( (answer - y)^2 )
        # loss = np.sum(np.sqrt((ground_truth - output) ** 2))
        loss = ground_truth - output

        # Back propagate the weights
        dw_2 = np.dot(loss, sigmoid_derivative(output))

        l1_loss = np.dot(dw_2, w_1.T)
        dw_1 = np.dot(l1_loss, sigmoid_derivative(l1))

        # Nudge the weights in the wright direction
        w_2 += np.dot(l1.T, dw_2)
        w_1 += np.dot(input.T, dw_1)

        if i % 10000 == 0:
            print("Loss: ", np.mean(np.abs(loss)))

print(output)