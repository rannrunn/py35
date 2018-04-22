# https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f
# https://github.com/devnag/pytorch-generative-adversarial-networks/blob/master/gan_pytorch.py
# (cuda) https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/
# 02-intermediate/convolutional_neural_network/main-gpu.py


# todo: refactoring codes below

from optimization import pso_classVer as pso
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from preprocessing import preprocessing as pre

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import copy
import random


def generator(layers_):
    legos = []
    for idx in range(len(layers_) - 1):
        in_n = layers_[idx]
        out_n = layers_[idx + 1]

        # linear sum
        legos.append(nn.Linear(in_n, out_n))

        if idx != (len(layers_) - 2):  # range: -1 & in_n: -1, therefore: -2
            # act. func.
            legos.append(nn.ELU())

    # output layer
    legos.append(nn.Tanh())

    _model = nn.Sequential(*legos)
    return _model


def discriminator(layers_):
    legos = []
    for idx in range(len(layers_) - 1):
        in_n = layers_[idx]
        out_n = layers_[idx + 1]

        # linear sum
        legos.append(nn.Linear(in_n, out_n))

        if idx != (len(layers_) - 2):  # range: -1 & in_n: -1, therefore: -2
            # act. func.
            legos.append(nn.ELU())

    # output layer
    legos.append(nn.Sigmoid())
    _model = nn.Sequential(*legos)
    return _model

# input_size = 3
# hidden_size = 5
# output_size = 2


def print_metric(D, real_data, gen_data):
    print("D prob (on real data): ", round(torch.mean(D(real_data).data), 3))
    print("D prob (on gen data): ", round(torch.mean(D(gen_data).data), 3))


if __name__ == "__main__":
    def loss_fn(args):
        _sum = 0
        for val in args:
            _sum += abs(val)
        return _sum


    def get_init_param(min_=0, max_=100, dim=2):
        _params = [None for _ in range(dim)]
        for _idx in range(dim):
            _params[_idx] = random.randint(min_, max_)
        return np.array(_params)


    def is_terminated(count):
        if count > 300:
            return True
        else:
            return False

    # init
    col_num = 2

    # set hyper parameter
    hp = pso.PsoHyperParam(inertia_=0.4, c1_=2, c2_=2, population_size_=10)
    global_param = pso.get_opt_param(hp=hp, loss_fn=loss_fn, get_init_param_fn=get_init_param,
                                     is_terminated_fn=is_terminated, verbose=True)

    print(global_param.param)
    print(global_param.l_loss)


    # --- data --- #
    # --- 1. normal (simul)     # simul: simulation
    data = pre.get_normal_data(mu_=0, sigma_=1, n_=1000, dim_=2, is_return_tensor_var=True)
    data = data.type(torch.FloatTensor)

    # --- generator & discriminator --- #
    G_layers = [100, 200, 50, 2]
    G = generator(G_layers)

    D_layers = [2, 5, 3, 1]
    D = discriminator(D_layers)

    if torch.cuda.is_available():
        D.cuda()
        G.cuda()

    # --- optimizer --- #
    # todo: study Adam optimizer
    learning_rate = 0.0001

    d_lr, d_betas, weight_decaying = learning_rate, (0.9, 0.5), 0.01
    d_optimizer = optim.Adam(D.parameters(), lr=d_lr, betas=d_betas, weight_decay=weight_decaying)

    g_lr, g_betas = learning_rate, (0.9, 0.5)
    g_optimizer = optim.Adam(G.parameters(), lr=g_lr, betas=g_betas, weight_decay=weight_decaying)

    # --- parameters --- #
    epochs, d_epochs, g_epochs = 100, 1, 1
    n_row = data.size()[0]
    batch_size = 100
    total_batch = n_row//batch_size

    d_errors = [None for _ in range(epochs * d_epochs * total_batch)]
    g_errors = [None for _ in range(epochs * g_epochs * total_batch)]
    d_probs = [None for _ in range(epochs * d_epochs * total_batch)]
    g_probs = [None for _ in range(epochs * g_epochs * total_batch)]

    # todo: d 를 속이기 너무 쉬운게 문제 아닐까...?
    # todo: d 를 다른 방식으로 학습 시켜서 ㅡ gen 학습시켜봄
    # todo: classification boundary 도 그려봄
    # todo: 예를 들어 가짜 데이터 만들어서 ㅡ 초기 학습
    # todo: 목적함수도 다르게함
    # todo: 목적함수 그림 그려봄

    for epoch in tqdm(range(epochs)):
        # todo: shuffle 할 때 왜 데이터 겹치는 문제가 생길까...?
        # data = data[np.random.randint(0, n_row, size=n_row), :]   # shuffle

        # if epochs % 100 == 0:
        #     learning_rate *= 0.1        # learning rate scheduling

        for d_epoch in range(d_epochs):
            for batch_idx in range(total_batch):
                # make generated data
                z_noise = pre.get_normal_data(mu_=0, sigma_=1, n_=batch_size, dim_=G_layers[0])
                z_noise = Variable(torch.from_numpy(z_noise), requires_grad=True).type(torch.FloatTensor)
                z_noise = z_noise.cuda()
                gen_data = G(z_noise)
                real_data = data[(batch_size * batch_idx):(batch_size * (batch_idx+1)), :]
                real_data = real_data.cuda()

                d_error = torch.log(D(real_data)) + torch.log(1-D(gen_data))
                d_error = -d_error  # to ascend gradients
                d_error = torch.mean(d_error)

                d_errors[epoch * total_batch + batch_idx] = d_error.data[0]
                d_error.backward()
                d_optimizer.step()
                D.zero_grad()
            # print_metric(D, real_data, gen_data)

        for g_epoch in range(g_epochs):
            for batch_idx in range(total_batch):
                # make generated data
                z_noise = pre.get_normal_data(mu_=0, sigma_=1, n_=batch_size, dim_=G_layers[0])
                z_noise = Variable(torch.from_numpy(z_noise), requires_grad=True).type(torch.FloatTensor)
                z_noise = z_noise.cuda()
                gen_data = G(z_noise)

                # todo: mnist 에 사용된 학습 방법 이해 & 사용해봄...?
                g_error = -torch.log(D(gen_data))       # g_error = torch.log(1-D(gen_data))
                g_error = torch.mean(g_error)
                g_errors[epoch * total_batch + batch_idx] = g_error.data[0]
                g_error.backward()
                g_optimizer.step()
                G.zero_grad()

                g_probs[epoch * total_batch + batch_idx] = round(torch.mean(D(gen_data).data), 3)
                d_probs[epoch * total_batch + batch_idx] = round(torch.mean(D(real_data).data), 3)
        # print_metric(D, real_data, gen_data)

        if epoch % 49 == 0:
            print("\n\nepoch: ", epoch)
            print_metric(D, real_data, gen_data)

# real data
# # 1) iris
# origin_data = pre.get_iris()
# origin_data = origin_data.iloc[:, 0:2]
# scatter_matrix(origin_data, alpha=0.2, figsize=(6, 6), diagonal='kde')


# 2. normal simulation data
scatter_matrix(pd.DataFrame(data.data.numpy()), alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.show()
plt.close()

# gen data
z_noise = pre.get_normal_data(mu_=0, sigma_=1, n_=10000, dim_=G_layers[0], is_return_tensor_var=True).type(torch.FloatTensor)
z_noise = z_noise.cuda()
gen_data = G(z_noise)

scatter_matrix(pd.DataFrame(np.asarray(gen_data.data)), alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.show()
plt.close()


# evaluate metric
plt.plot(d_probs, 'r')
plt.plot(g_probs, 'b')
plt.show()
plt.close()

plt.plot(d_errors, 'r')
plt.plot(g_errors, 'b')
plt.show()
plt.close()





