import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

x = Variable(torch.Tensor([0.4]))
y = Variable(torch.Tensor([1]))

loss_fn = nn.BCELoss()

loss = loss_fn(x, y)

print(loss.data[0])

m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.autograd.Variable(torch.randn(3), requires_grad=True)
target = torch.autograd.Variable(torch.FloatTensor(3).random_(2))
output = loss(m(input), target)
output.backward()

print(m(input).data[0])
print(target.data[0])
print(output.data[0])


