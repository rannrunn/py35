from matplotlib import pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.1, 0.1)
y = sigmoid(x)

print(x)
print(y)
print(np.exp(-x))
print(np.exp(-0))

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()



