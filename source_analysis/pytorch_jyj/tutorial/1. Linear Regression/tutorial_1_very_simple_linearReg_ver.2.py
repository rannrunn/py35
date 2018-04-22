# https://stackoverflow.com/questions/46513276/pytorch-backprop-through-volatile-variable-error?rq=1
"""
# simple linear regression

3 = w * 1

[change loss function]

before:
    loss = (y-y_hat).pow(2).sum()

after:
    loss_fn = nn.MSELoss()
    loss = loss_fn(input_, target)
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt

input_ = Variable(torch.Tensor([1]), requires_grad=True)
input_.retain_grad()
target = Variable(torch.Tensor([100]))

loss_fn = nn.MSELoss()

epochs = 1000
grads = [None for i in range(epochs)]
for epoch in range(epochs):
    loss = loss_fn(input_, target)
    loss.backward()
    grads[epoch] = input_.grad.data[0]

    input_.data = input_.data - input_.grad.data * 0.01
    input_.grad.data.zero_()

    if epoch % 500 == 0:
        print("\nloss:  ", round(loss.data[0], 3), "\n")

plt.plot(grads, 'ro')
plt.show()

