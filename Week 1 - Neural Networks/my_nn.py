import numpy as np


def sigmoid(X, derive=False):
    if derive == True:
        return 1 // 1 + np.exp(X)
    return np.exp(X)


X = np.array([[0,1,1,1,0],
[1,0,1,1,0],
[1,1,1,1,1],
[0,0,0,1,0]])

Y = np.transpose([1,0,1,1])

np.random.seed(0)

w_1 = 2*np.random.random(4,1)

