# http://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_nn.html
# https://github.com/yunjey/pytorch-tutorial
"""
tutorial_2_classify_iris
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from preprocessing import preprocessing as pre
import pandas as pd
import numpy as np
import torch.nn.functional as F


def build_structure(layers):
    legos = []
    for idx in range(len(layers) - 1):
        in_n = layers[idx]
        out_n = layers[idx + 1]

        # linear sum
        legos.append(nn.Linear(in_n, out_n))

        if idx != (len(layers) - 2):  # range: -1 & in_n: -1, therefore: -2
            # act. func.
            legos.append(nn.ReLU())

    # output layer
    legos.append(nn.Softmax(dim=1))
    _model = nn.Sequential(*legos)
    return _model


if __name__ == "__main__":
    # --- data --- #
    data = pre.get_iris()
    split = pre.SplitData(data_=data)
    data_x, data_y = split.x_y(y_col="Species")
    data_x_sc, scaler = pre.scale_0_to_1(data_x)    # sc: scaled

    data_y = data_y.astype("category")
    data_y_onehot = pd.get_dummies(data_y)

    # --- df to torch --- 3
    data_x_tensor = pre.df_to_torch(data_x_sc, requires_grad_=False)
    data_y = pre.df_to_torch(data_y, requires_grad_=False)

    # --- model --- #
    """
    N: batch size
    D_in: input dimension
    H: hidden dimension
    D_out: output dimension
    """
    N = 150
    # D_in, D_out = 4, 3
    # H1, H2 = 5, 4

    # model = nn.Sequential(
    #     nn.Linear(D_in, H1),
    #     nn.ReLU(),
    #     nn.Linear(H1, H2),
    #     nn.ReLU(),
    #     nn.Linear(H2, D_out),
    #     nn.Softmax(dim=1)
    # )

    layers = [4, 5, 4, 3]
    model = build_structure(layers)

    """
    [torch.nn.softmax parameter]
    dim (int) – A dimension along which Softmax 
    will be computed (so every slice along dim will sum to 1).

    # https://stackoverflow.com/questions/48070505/pytorch-softmax-dimensions-error
    """

    loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    lr = 1e-2

    for epoch in range(10000):
        softmax_output = model(data_x_tensor)
        y_pred = torch.max(softmax_output, 1)[1]

        data_y = data_y.type(torch.LongTensor)
        loss = loss_fn(softmax_output, data_y)

        model.zero_grad()   # set gradients as 0
        loss.backward()     # compute gradient of the loss

        # apply backpropagation (way 1)
        optimizer.step()

        # apply backpropagation (way 2)
        # for param in model.parameters():
        #     param.data -= lr * param.grad.data

        if epoch % 100 == 0:
            print(round(loss.data[0], 2))

    res = pd.DataFrame(np.vstack([y_pred.data.numpy(), data_y.data.numpy()]).T)     # res: result
    res.ix[0:50, :]
    res.ix[50:100, :]
    res.ix[100:, :]


