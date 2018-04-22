# https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f
# https://github.com/devnag/pytorch-generative-adversarial-networks/blob/master/gan_pytorch.py
# (cuda) https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/
# 02-intermediate/convolutional_neural_network/main-gpu.py


# todo: refactoring codes below

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


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        return F.tanh(self.map3(x))


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        return F.sigmoid(self.map3(x))


if __name__ == "__main__":
    # --- data --- #
    data = pre.get_iris()
    data = data.iloc[:, 0:2]

    data, scaler = pre.scale_minus1_to_1(data_=data)
    # scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')
    # plt.show()
    # plt.close()
    data = pre.df_to_torch(data)

    # --- generator & discriminator --- #
    g_input_size, g_hidden_size, g_output_size = 10, 10, 2
    G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)

    d_input_size, d_hidden_size, d_output_size = 2, 5, 1
    D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)

    if torch.cuda.is_available():
        D.cuda()
        G.cuda()

    # --- optimizer --- #
    # todo: study Adam optimizer
    d_lr, d_betas = 0.0002, (0.9, 0.5)
    # d_optimizer = optim.Adam(D.parameters(), lr=d_lr, betas=d_betas)
    d_optimizer = optim.Adam(D.parameters(), lr=d_lr)

    g_lr, g_betas = 0.0002, (0.9, 0.5)
    # g_optimizer = optim.Adam(G.parameters(), lr=g_lr, betas=g_betas)
    g_optimizer = optim.Adam(G.parameters(), lr=g_lr)

    # --- parameters --- #
    epochs, d_epochs, g_epochs = 1000, 1, 5
    n_row = data.size()[0]
    batch_size = 10
    total_batch = n_row//batch_size

    d_errors = [None for _ in range(epochs * d_epochs * total_batch)]
    g_errors = [None for _ in range(epochs * g_epochs * total_batch)]
    d_probs = [None for _ in range(epochs * d_epochs * total_batch)]
    g_probs = [None for _ in range(epochs * g_epochs * total_batch)]

    for epoch in tqdm(range(epochs)):
    # for epoch in range(epochs):
        data = data[np.random.randint(0, n_row, size=150), :]   # shuffle
        for d_epoch in range(d_epochs):
            for batch_idx in range(total_batch):
                # make generated data
                z_noise = pre.get_normal_data(mu_=0, sigma_=1, n_=batch_size, dim_=g_input_size)
                z_noise = Variable(torch.from_numpy(z_noise), requires_grad=True).type(torch.FloatTensor)
                z_noise = z_noise.cuda()
                gen_data = G(z_noise)
                real_data = data[(batch_size * batch_idx):(batch_size * (batch_idx+1)), :]
                real_data = real_data.cuda()

                d_error = torch.log(D(real_data)) + torch.log(1-D(gen_data))
                d_error = torch.mean(d_error)
                d_error = -d_error      # to ascend gradients

                d_errors[epoch * total_batch + batch_idx] = d_error.data[0]
                d_error.backward()
                d_optimizer.step()
                D.zero_grad()

        for g_epoch in range(g_epochs):
            for batch_idx in range(total_batch):
                # make generated data
                z_noise = pre.get_normal_data(mu_=0, sigma_=1, n_=batch_size, dim_=g_input_size)
                z_noise = Variable(torch.from_numpy(z_noise), requires_grad=True).type(torch.FloatTensor)
                z_noise = z_noise.cuda()
                gen_data = G(z_noise)

                g_error = -torch.log(D(gen_data))  # g_error = torch.log(1-D(gen_data))
                g_error = torch.mean(g_error)
                g_errors[epoch * total_batch + batch_idx] = g_error.data[0]
                g_error.backward()
                g_optimizer.step()
                G.zero_grad()

                g_probs[epoch * total_batch + batch_idx] = round(torch.mean(D(gen_data).data), 3)
                d_probs[epoch * total_batch + batch_idx] = round(torch.mean(D(real_data).data), 3)

        if epoch % 100 == 0:
            print("\n\nepoch: ", epoch)
            print("discriminator error: ", round(d_error.data[0], 3))
            print("generator error: ", round(g_error.data[0], 3), "\n")

            print("D prob (on real data): ", round(torch.mean(D(real_data).data), 3))
            print("D prob (on gen data): ", round(torch.mean(D(gen_data).data), 3))


# real data
origin_data = pre.get_iris()
origin_data = origin_data.iloc[:, 0:2]
scatter_matrix(origin_data, alpha=0.2, figsize=(6, 6), diagonal='kde')

# gen data
z_noise = pre.get_normal_data(mu_=0, sigma_=1, n_=10000, dim_=g_input_size, is_return_tensor_var=True).type(torch.FloatTensor)
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
