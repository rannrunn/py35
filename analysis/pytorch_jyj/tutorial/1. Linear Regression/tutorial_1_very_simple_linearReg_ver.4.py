# simple linear regression

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import preprocessing as pre

"""
clipping gradient
"""

"""
clipping gradient works well (gradient explodint does not occur)
even if we choose less optimal paramter
but the result seems to be local-optimal

i.e. 
optimal paramter:      w=3,     b=2
near optimal paramter: w=3,     b=2.001 // learning_rate = 0.0001
but learned parameter: w= 2.998 b=2.158 // learning_rate = 0.001
but learned parameter: w= 2.981 b=3.45  // learning_rate = 0.01  
but learned parameter: w= 2.863 b=5.991 // learning_rate = 0.1

# https://www.coursera.org/learn/intro-to-deep-learning/lecture/Wes8G/dealing-with-vanishing-and-exploding-gradients
"""


def clip_gradient(grad_data, clip_val=1):
    """
    grad: gradient
    val: value

    example:
    w.grad.data = clip_gradient(w.grad.data)
    """
    _grad_val = grad_data[0]
    _grad_norm = abs(_grad_val)

    if abs(_grad_val) > clip_val:
        _grad_val = clip_val * (_grad_val/_grad_norm)

    grad_data[0] = _grad_val
    return grad_data


n_row = 100
x = Variable(torch.Tensor(np.linspace(start=1, stop=100, num=n_row)), requires_grad=False)
# noise = pre.get_normal_data(mu_=0, sigma_=1, n_=n_row, dim_=1, is_return_tensor_var=True)
# noise = noise.type(torch.FloatTensor)
# noise = noise.view(-1)  # reshape [100,1] → [100]

# y = 3*x + 2 + noise
# todo: noise 있으면 왜 local minima 에 빠져서 못나올까...?
y = 3*x + 2

# plt.plot(np.asarray(y.data))
# plt.show()

w = Variable(torch.Tensor([0]), requires_grad=True)
b = Variable(torch.Tensor([0]), requires_grad=True)
w.retain_grad()     # to record 'w.grad'
b.retain_grad()     # to record 'b.grad'

# optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

epochs = 10000
grads_w = [None for i in range(n_row)]
grads_b = [None for i in range(n_row)]
losses = [None for i in range(n_row)]

epoch = 0
row = -1


# todo: 왜 학습이 안되지...?
for epoch in range(epochs):
    for row in range(n_row):

        y_hat = w * x[row] + b
        loss = (y[row]-y_hat).pow(2)       # loss = (y-y_hat)**2

        loss.backward()

        learning_rate = 0.001      # 0.0001 is good

        # update paramters
        w.grad.data = clip_gradient(w.grad.data, clip_val=1)
        b.grad.data = clip_gradient(b.grad.data, clip_val=1)

        w.data = w.data - learning_rate * w.grad.data
        b.data = b.data - learning_rate * b.grad.data

        # record gradients (&loss)
        grads_w[row] = w.grad.data[0]
        grads_b[row] = b.grad.data[0]
        losses[row] = loss

        # set 'gradients' as '0'
        w.grad.data.zero_()
        b.grad.data.zero_()

        if epoch % 1000 == 0 and row == 0:
            # print("\nw:     ", round(w.data[0], 3), "\n")
            print("\nloss:  ",  round(loss.data[0], 3),
                  # "\ny_hat: ", round(y_hat.data[0], 3),
                  "\nb:     ", round(b.data[0], 3),
                  "\nw:     ", round(w.data[0], 3), "\n")
            print("\nepoch: ", epoch, "\n")

plt.plot(grads_w, 'ro', label="weight")
plt.plot(grads_b, 'bo', label="bias")
plt.plot(losses, 'go', label="loss")
plt.legend(loc='best')
plt.show()

