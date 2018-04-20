
"""
# simple linear regression

y = 3 * x + 2


"""


import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import plot

"""
Problem. 1:
if we add noise to y, 
optimal weight could not be found  

Problem. 2:
if we raise learning rate above 0.0001 such as 0.01
gradients explode 

these problems will be addressed after this tutorial 
"""


n_row = 100
x = Variable(torch.Tensor(np.linspace(start=1, stop=100, num=n_row)), requires_grad=False)
y = 3*x + 2

# plt.plot(np.asarray(y.data))
# plt.show()

w = Variable(torch.Tensor([0]), requires_grad=True)
b = Variable(torch.Tensor([0]), requires_grad=True)
w.retain_grad()     # to record 'w.grad'
b.retain_grad()     # to record 'b.grad'

grads_w = []
weights = []
grads_b = []
biases = []
losses = []

epochs = 1000
row = 0
pre_loss = 0
if __name__ == "__main__":

    for epoch in range(epochs):
        for row in range(100):

            # if row == 149:
            #     row = 0
            #

            y_hat = w * x[row] + b
            loss = (y[row]-y_hat).pow(2)       # loss = (y-y_hat)**2

            loss.backward()

            print(w, b)
            print(w.grad, b.grad)
            print(pre_loss, loss)

            learning_rate = 0.001  # problem 2.    # near optimal parameter: 0.0001

            # update paramters
            w.data = w.data - learning_rate * w.grad.data
            b.data = b.data - learning_rate * b.grad.data

            # --- record parameters & gradients &loss
            # weights.append(w.data[0])
            # biases.append(b.data[0])
            # grads_w.append(w.grad.data[0])
            # grads_b.append(b.grad.data[0])
            # losses.append(loss.data[0])

            # set 'gradients' as '0'
            w.grad.data.zero_()
            b.grad.data.zero_()
            pre_loss = loss
            row += 1
            print("\nloss:  ",  round(loss.data[0], 3),
                  "\ny_hat: ", round(y_hat.data[0], 3),
                  "\nb:     ", round(b.data[0], 3),
                  "\nw:     ", round(w.data[0], 3), "\n")

    plt.plot(grads_w, 'ro', label="weight")
    plt.plot(grads_b, 'bo', label="bias")
    plt.legend(loc='best')
    plt.show()
    plt.close()

    plt.plot(losses, 'go', label="loss")
    plt.show()
    plt.close()

    # --- plot loss surface --- #
    n = 2000
    weights_ = np.linspace(min(weights), max(weights), n)
    biases_ = np.linspace(min(biases), max(biases), n)
    losses_ = [None for _ in range(n)]

    for idx in range(n):
        w = weights_[idx]
        b = biases_[idx]

        y_hat = w * x.data + b
        loss = (y.data - y_hat).pow(2).mean()  # loss = (y-y_hat)**2
        losses_[idx] = loss

    plot.contour_3d(weights_, biases_, losses_)

    # --- visualizing loss --- #
    w_direction = weights[len(weights)-1] - weights[0]
    b_direction = biases[len(weights)-1] - biases[0]

    min_idx = losses.index(min(losses))
    w_center = weights[min_idx]
    b_center = biases[min_idx]

    n = 100
    weights_ = np.linspace(w_center - n * w_direction, w_center + n * w_direction, n)
    biases_ = np.linspace(b_center - n * b_direction, b_center + n * b_direction, n)
    losses_ = [None for _ in range(n)]

    for idx in range(n):
        w = weights_[idx]
        b = biases_[idx]

        y_hat = w * x.data + b
        loss = (y.data - y_hat).pow(2).mean()  # loss = (y-y_hat)**2
        losses_[idx] = loss

    plot.contour_3d(weights_, biases_, losses_)

    _fig = pylab.figure()
    _ax = Axes3D(_fig)
    _ax.plot_surface(weights, biases, losses, cmap="autumn_r", lw=0, rstride=1, cstride=1)


