import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

y = Variable(torch.Tensor([4]))

list = np.arange(0, 8, 0.01)
list_result = []

for item in list:
    list_result.append((y - item).pow(2).data[0])

print(list_result)
plt.plot(list, list_result, 'r')
plt.show()