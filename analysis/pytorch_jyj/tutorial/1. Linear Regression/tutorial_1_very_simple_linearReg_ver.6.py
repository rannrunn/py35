# simple linear regression

# todo: gradient 가 파도 타기 하면서, loss 0 까지 가다가 ㅡ 움직여서 커질 수 있음 (시우쌤)
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
import pylab

from preprocessing import plot
import numpy as np
import copy
from preprocessing import preprocessing as pre

"""
clipping gradient & mini-Batch Learning 
"""

"""
clipping gradient works well (gradient explodint does not occur)
even if we choose less optimal paramter
but the result seems to be local-optimal

> As observed by Keskar et al. (2017), we can clearly see that the small-batch solution is quite wide,
> while the large-batch solution is sharp.
> "Visualizing the loss landscape of neural nets"

i.e. 
optimal paramter:      w=3,     b=2
near optimal paramter: w=3,     b=2.001             // learning_rate = 0.0001

but learned parameter: w= 2.998 b=2.158             // learning_rate = 0.001
>  mini-batch (n: 5):  w= 2.999 b=2.078             // learning_rate = 0.001    ★
>  mini-batch (n: 5) + shuffle:  w= 1.796 b=39.182  // learning_rate = 0.001
> mini-batch (n: 10):  w= 2.983 b=2.979             // learning_rate = 0.001 
> mini-batch (n: 10) + shuffle:  w= 1.994 b=23.12   // learning_rate = 0.001   

but learned parameter: w= 2.981 b=3.45              // learning_rate = 0.01     ★
>  mini-batch (n: 5):  w= 1.751 b=57.608            // learning_rate = 0.01
>  mini-batch (n: 5) + shuffle:  w= 0.342 b=138.076 // learning_rate = 0.01 
> mini-batch (n: 10):  w= 2.15  b=28.366            // learning_rate = 0.01     
> mini-batch (n: 10) + shuffle:  w= 0.74  b=109.379 // learning_rate = 0.01 

but learned parameter: w= 2.863 b=5.991             // learning_rate = 0.1      ★
>  mini-batch (n: 5):  w= 2.7   b=13.8              // learning_rate = 0.1 

# https://www.coursera.org/learn/intro-to-deep-learning/lecture/Wes8G/dealing-with-vanishing-and-exploding-gradients
"""

# todo: idea! plotting with RL method (i.e. monte-carlo search...?)

# todo: interpret above results according to parameters
# todo: mini-batch results in better result. why...?
# todo: shuffle results in worse result. why...?

# todo: experiment - weight decay


def clip_gradient(grad_data, clip_val=1):
    # todo: idea! clip scheduling...? research & experiment & write paper
    """
    grad: gradient
    val: value

    example:
    w.grad.data = clip_gradient(w.grad.data)
    """
    _grad_val = grad_data[0]
    _grad_norm = abs(_grad_val)

    if abs(_grad_val) > clip_val:
        _grad_val = (clip_val / _grad_norm) * _grad_val

    grad_data[0] = _grad_val
    return grad_data


n_row = 100
x = Variable(torch.Tensor(np.linspace(start=1, stop=100, num=n_row)), requires_grad=False)
# noise = pre.get_normal_data(mu_=0, sigma_=1, n_=n_row, dim_=1, is_return_tensor_var=True)
# noise = noise.type(torch.FloatTensor)
# noise = noise.view(-1)  # reshape [100,1] → [100]

# y = 3*x + 2 + noise
# todo: noise 있으면 왜 local minima 에 빠져서 못나올까...?
y = 3 * x + 2

# plt.plot(np.asarray(y.data))
# plt.show()

w = Variable(torch.Tensor([0]), requires_grad=True)
b = Variable(torch.Tensor([0]), requires_grad=True)
w.retain_grad()  # to record 'w.grad'
b.retain_grad()  # to record 'b.grad'

# optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

epochs = 5000
learning_rate = 0.1  # 0.0001 is good
# todo: 왜 learing_rate 높으면 ㅡ 어느정도 수렴되다가 ㅡ exploding 할까...?
# todo: exploding 은 왜 일어날까...?
batch_size = 5
is_shuffle = False
use_clipping = True
batch_num = n_row // batch_size
count = -1

grads_w = [None for i in range(batch_num * epochs)]
grads_b = [None for i in range(batch_num * epochs)]
losses = [None for i in range(batch_num * epochs)]
weights = [None for i in range(batch_num * epochs)]
biases = [None for i in range(batch_num * epochs)]
learning_rates = [None for i in range(batch_num * epochs)]

is_lr_decaying = True       # lr: learning rate

for epoch in range(epochs):
    if is_lr_decaying:
        if epoch % 1000 == 0:
            learning_rate *= 0.1

    # shuffle x
    x_shuffle = copy.deepcopy(x)
    if is_shuffle:
        shuffle_idx = np.random.choice(n_row, n_row, replace=False).tolist()
        x_shuffle = x_shuffle[shuffle_idx]

    for b_idx in range(batch_num):
        """
        b_idx: batch_idx
        s: start
        e: end  
        """
        s = 0 + batch_size * b_idx
        e = s + batch_size

        count += 1

        y_hat = w * x_shuffle[s:e] + b
        loss = (y[s:e] - y_hat).pow(2).mean()  # loss = (y-y_hat)**2

        loss.backward()

        # update paramters
        if use_clipping:
            w.grad.data = clip_gradient(w.grad.data, clip_val=1)
            b.grad.data = clip_gradient(b.grad.data, clip_val=1)  # recommend: clip_val=0.1

        w.data = w.data - learning_rate * w.grad.data  # todo: 왜 gradient of w 가 폭발하지...?
        b.data = b.data - learning_rate * b.grad.data

        # record parameters
        weights[count] = w.data[0]
        biases[count] = b.data[0]
        learning_rates[count] = learning_rate

        # record gradients (&loss)
        grads_w[count] = w.grad.data[0]
        grads_b[count] = b.grad.data[0]

        losses[count] = loss.data[0]
        # todo: sgd 문제 공부, adam 공부
        # todo: pytorch 기본 optim 써서 업데이팅 시키는 방법...?

        # set 'gradients' as '0'
        w.grad.data.zero_()
        b.grad.data.zero_()

        if epoch % 100 == 0 and b_idx == 0:
            # print("\nw:     ", round(w.data[0], 3), "\n")
            print("\nloss:  ", round(loss.data[0], 3),
                  # "\ny_hat: ", round(y_hat.data[0], 3),
                  "\nb:     ", round(b.data[0], 3),
                  "\nw:     ", round(w.data[0], 3), "\n")
            print("\nlearning rate: ", learning_rate, "\n")
            print("\nepoch: ", round(epoch, 3), "\n")


# --- plot --- #
# plt.plot(losses, 'go', label="loss")
# plt.plot(grads_w, 'ro', label="weight")
# plt.plot(grads_b, 'bo', label="bias")
plt.plot(weights, 'ro', label="weight")
plt.plot(biases, 'bo', label="bias")
plt.legend(loc='best')
plt.show()
plt.close()

plt.plot(learning_rates, 'go', label="learning rate")
plt.legend(loc='best')
plt.show()
plt.close()

# --- 3d plot --- #
idx = 100000
xx, yy, zz = weights[-idx:], biases[-idx:], losses[-idx:]

fig = pylab.figure()
ax = Axes3D(fig)
ax.scatter(xx, yy, zz)
ax.plot(xx, yy, zz)
ax.set_xlabel('weights')
ax.set_ylabel('biases')
ax.set_zlabel('losses')
pyplot.show()
# pyplot.close()


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

# plot.contour_2d(weights_, biases_, losses_)
plot.contour_3d(weights_, biases_, losses_)
