# https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/302_classification.py
# https://discuss.pytorch.org/t/how-to-calculate-gradient-for-each-layer/1595/5

"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.1.11
matplotlib
"""


import torch
from torch.autograd import Variable
import torch.nn.functional as F     # for activation funcs
import matplotlib.pyplot as plt


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()

        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x_):
        x_ = F.relu(self.hidden(x_))      # activation function for hidden layer
        res = self.out(x_)
        return res


def get_x_y():
    # --- simulation data --- #
    ones = torch.ones(100, 2)
    x0 = torch.normal(2 * ones, 1)  # class0 x data (tensor), shape=(100, 2)
    y0 = torch.zeros(100)  # class0 y data (tensor), shape=(100, 1)
    x1 = torch.normal(-2 * ones, 1)  # class1 x data (tensor), shape=(100, 2)
    y1 = torch.ones(100)  # class1 y data (tensor), shape=(100, 1)
    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
    y = torch.cat((y0, y1), ).type(torch.LongTensor)  # shape (200,)   LongTensor = 64-bit integer

    # --- convert tensor to Variable --- #
    x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)
    return x, y


if __name__ == "__main__":
    x, y = get_x_y()

    # --- Build Networks --- #
    net = Net(n_feature=2, n_hidden=3, n_output=2)     # define the network
    print(net)  # net architecture

    # --- set optimizer & loss function --- #
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

    # --- predict & reset parameters & plot --- #
    epochs = 1000
    for epoch in range(epochs):

        # --- predict --- #
        out = net(x)                 # input x and predict based on x

        # --- calculate loss --- #
        loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

        # --- calculate gradient --- #
        loss.backward()         # compute gradients

        # --- see gradients of weights --- #
        grad_of_params = {}
        for name, parameter in net.named_parameters():
            grad_of_params[name] = parameter.grad

        # --- set gradient as 0 --- #
        optimizer.zero_grad()   # clear gradients for next train

        # --- apply gradient --- #
        optimizer.step()        # apply gradients

        if epoch % 100 == 0:
            print(round(loss.data[0], 3))
