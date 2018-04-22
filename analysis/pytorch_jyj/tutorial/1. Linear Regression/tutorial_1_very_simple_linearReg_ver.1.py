
"""
# simple linear regression

3 = w * 1
"""
import time
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


y = Variable(torch.Tensor([4]), requires_grad=False)
x = Variable(torch.Tensor([1]), requires_grad=False)
w = Variable(torch.Tensor([0]), requires_grad=True)
w.retain_grad()     # to record 'w.grad'

# time.sleep(1000)

y_hat = w * x

epochs = 410
grads = [None for i in range(epochs)]
losses = [None for i in range(epochs)]
epoch = 0

for epoch in range(epochs):
    loss = (y - y_hat).pow(2)       # loss = (y - y_hat)**2
    loss.backward()

    learning_rate = 0.01
    w.data = w.data - learning_rate * w.grad.data
    grads[epoch] = round(w.grad.data[0], 3)
    losses[epoch] = loss.data[0]

    w.grad.data.zero_()     # set 'gradient of w' as '0'

    y_hat = w * x
    if epoch % 1 == 0 or epoch == epochs - 1:
        print("\nloss:  ",  round(loss.data[0], 3),
              "\ny_hat: ", round(y_hat.data[0], 3),
              "\nw:     ", round(w.data[0], 3), "\n")

print(len(grads))
print(losses[0])
print(losses[1])
print(losses[2])
print(grads)
plt.plot(grads, 'ro')
plt.show()

