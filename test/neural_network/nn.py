import numpy as np
from matplotlib import pyplot as plt

def softmax(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cross_entropy_error(y, t):
    return -np.sum(t * np.log(y))

def mean_squared_error(y, t):
    return (1/2)(np.sum(np.square((y, t))))

input = [[1,2,3],[4,5,6]]

W1 = [[0,0],[0,0],[0,0]]
b1 = [[0,0],[0,0]]

h1 = np.dot(input, W1)

output = np.dot(h1, b1)

output = []






