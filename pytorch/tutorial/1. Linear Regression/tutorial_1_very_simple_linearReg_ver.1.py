
"""
# simple linear regression

3 = w * 1
"""

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

y = Variable(torch.Tensor([3]), requires_grad=False)

x = Variable(torch.Tensor([1]), requires_grad=False)
w = Variable(torch.Tensor([0]), requires_grad=True)
w.retain_grad()     # to record 'w.grad'

y_hat = w*x

epochs = 5000
grads = [None for i in range(epochs)]
epoch = 0

for epoch in range(epochs):
    loss = (y-y_hat).pow(2).sum()       # loss = (y-y_hat)**2
    loss.backward()

    learning_rate = 0.001
    w.data = w.data - learning_rate * w.grad.data
    grads[epoch] = w.grad.data[0]
    w.grad.data.zero_()     # set 'gradient of w' as '0'

    y_hat = w*x
    if epoch % 1000 == 0:
        print("\nloss:  ",  round(loss.data[0], 3),
              "\ny_hat: ", round(y_hat.data[0], 3),
              "\nw:     ", round(w.data[0], 3), "\n")

plt.plot(grads, 'ro')
plt.show()

