from matplotlib import pyplot as plt
import numpy as np

def relu(x):
    return np.maximum(0, x)

x = np.arange(-15.0, 15.1, 0.1)
y = relu(x)

print(x)
print(y)

plt.plot(x, y)
plt.show()