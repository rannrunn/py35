import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable


y = Variable(torch.Tensor([4]).cuda(), requires_grad=False)
w = Variable(torch.Tensor([0]).cuda(), requires_grad=True)
x = Variable(torch.Tensor([1]).cuda(), requires_grad=False)
w.retain_grad()


epochs = 2000
learning_rate = 0.001
weights = []
grads = []
losses = []


x_loss = np.arange(0, 8, 0.01)
y_loss = []
for item in x_loss:
    y_loss.append((y - item).pow(2).data)

plt.plot(y_loss, 'r')
plt.show()


for epoch in range(epochs):
    loss = (y - w * x).pow(2)
    loss.backward()

    w.data = w.data - learning_rate * w.grad.data

    weights.append(round(w.data[0], 3))
    grads.append(round(w.grad.data[0], 3))
    losses.append(round(loss.data[0], 3))

    w.grad.data.zero_()


plt.plot(weights, 'r')
plt.show()
plt.plot(grads, 'b')
plt.show()
plt.plot(losses, 'g')
plt.show()

