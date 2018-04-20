
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable


# --- 1. np array example --- #
mu, sigma, n, dim = 1, 0.5, 1000, 2

data = np.random.normal(mu, sigma, (n, dim))

plt.plot(data[:, 0], data[:, 1], 'go')
plt.show()
plt.close()

# --- 2. np array -> torch tensor --- #
"""
torch tensor is similar to np array
but GPU calculation is possible
"""
data = torch.from_numpy(data)
data = data.type(torch.DoubleTensor)    # set torch type
type(data)
data.size()
# data = torch.from_numpy(data).type(torch.DoubleTensor)
# http://pytorch.org/docs/master/tensors.html



# --- 3. torch tensor -> torch Variable --- #
"""
Variable is needed to do 'backprop'

(backprop is addressed after tutorial_0)
"""
torch_var = Variable(data, requires_grad=False)     # does not calculate gradient, i.e. input data
type(torch_var)
torch_var.requires_grad
torch_var.data          # get data
torch_var.grad          # see grad
torch_var.grad.data     # get grad (if gradient exists)
torch_var.grad_fn       # see grad func (not used frequently...?!)

torch_var = Variable(data, requires_grad=True)      # calculate gradient, i.e. hidden weights
type(torch_var)
torch_var.requires_grad
torch_var.data          # get data
torch_var.grad          # see grad
torch_var.grad.data     # get grad (if gradient exists)
torch_var.grad_fn       # see grad func (not used frequently...?!)

# --- 4. other ways to make torch tensor --- #
# 4-1) torch.Tensor
data = torch.Tensor([3])
type(data)
data.size()

data = torch.Tensor([3, 4])
type(data)
data.size()

# 4-2) torch.randn  (random samples from normal dist.)
n, dim, dtype = 1000, 2, torch.DoubleTensor
data = torch.randn(n, dim).type(dtype)
type(data)
data.size()

data = np.asarray(data)     # tensor torch -> np array
plt.plot(data[:, 0], data[:, 1], 'go')
plt.show()


# --- 5. resize --- #

# import torch
t = torch.ones((2, 3))
t
t.size()
t.view(3, 2)
t.view(3, 2).size()


