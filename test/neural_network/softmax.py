import numpy as np
from matplotlib import pyplot as plt

def softmax(x):
    max = np.max(x)
    return np.exp(x - max) / np.sum(np.exp(x - max))

def softmax_2(x):
    return np.exp(x) / np.sum(np.exp(x))

arr = np.array([0.3, 2.9, 4.0])

print(arr / 7.3)
print(softmax(arr))
print(softmax(arr))
print(arr - np.max(arr))

