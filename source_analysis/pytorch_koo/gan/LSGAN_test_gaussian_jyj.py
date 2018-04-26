# https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f
# https://github.com/devnag/pytorch-generative-adversarial-networks/blob/master/gan_pytorch.py
# (cuda) https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/
# 02-intermediate/convolutional_neural_network/main-gpu.py
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import pandas as pd


def plot_gan_result(data, gen_data):
    origin_data = pd.DataFrame(data.data.numpy())
    origin_data['class'] = 1
    generated_data = pd.DataFrame(np.asarray(gen_data.data))
    generated_data['class'] = 0
    combined_data = origin_data.append(generated_data)
    plt.scatter(combined_data.iloc[:, 0], combined_data.iloc[:, 1], s=100, marker='*', c=combined_data['class'])
    # plt.show()


def generator(layers_):
    legos = []
    for idx in range(len(layers_) - 1):
        in_n = layers_[idx]
        out_n = layers_[idx + 1]

        # linear sum
        legos.append(nn.Linear(in_n, out_n))

        if idx != (len(layers_) - 2):  # range: -1 & in_n: -1, therefore: -2
            # act. func.
            legos.append(nn.Dropout(p=0))
            legos.append(nn.BatchNorm1d(out_n))
            legos.append(nn.LeakyReLU(0.2))

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
            legos.append(nn.Dropout(p=0))
            legos.append(nn.BatchNorm1d(out_n))
            legos.append(nn.LeakyReLU(0.2))

    # output layer
    legos.append(nn.Sigmoid())

    _model = nn.Sequential(*legos)
    return _model


def print_metric(d_model, real_data_, gen_data_):
    print("\n\nD prob (on real data): ", round(torch.mean(d_model(real_data_).data), 3))
    print("D prob (on gen data): ", round(torch.mean(d_model(gen_data_).data), 3))


if __name__ == "__main__":
    # --- data --- #
    # 1) normal: unimodal
    n_row = 3000
    dim = 2
    data = torch.randn([n_row, dim])
    data = Variable(data, requires_grad=False).type(torch.FloatTensor)

    # --- generator & discriminator --- #
    g_input_size = 2
    G = generator([g_input_size, 10, 10, 2])
    D = discriminator([2, 10, 10, 1])

    # --- optimizer --- #
    learning_rate = 0.0003
    d_optimizer = optim.Adam(D.parameters(), lr=learning_rate)
    g_optimizer = optim.Adam(G.parameters(), lr=learning_rate)

    # --- parameters --- #
    epochs, d_epochs, g_epochs = 10, 1, 1
    n_row = data.size()[0]
    batch_size = 100
    total_batch = n_row // batch_size

    if torch.cuda.is_available():
        D.cuda()
        G.cuda()

    d_probs = [None for _ in range(epochs)]
    g_probs = [None for _ in range(epochs)]

    for epoch in tqdm(range(epochs)):
        data = data[np.random.choice(n_row, size=n_row, replace=False), :]  # shuffle

        for d_epoch in range(d_epochs):
            for batch_idx in range(total_batch):
                # make generated data
                z_noise = torch.randn([batch_size, dim])
                z_noise = Variable(z_noise, requires_grad=False).type(torch.FloatTensor).cuda()
                gen_data = G(z_noise)
                real_data = data[(batch_size * batch_idx):(batch_size * (batch_idx+1)), :]
                real_data = real_data.cuda()

                d_error = (1/2)*((D(real_data)-1)**2) + (1/2)*((D(gen_data)-0)**2)
                d_error = torch.mean(d_error)

                D.zero_grad()
                d_error.backward()
                d_optimizer.step()

        for g_epoch in range(g_epochs):
            for batch_idx in range(total_batch):
                # make generated data
                z_noise = torch.randn([batch_size, dim])
                z_noise = Variable(z_noise, requires_grad=False).type(torch.FloatTensor).cuda()
                gen_data = G(z_noise)

                g_error_tmp = (1/2)*((D(gen_data)-1)**2)

                g_error = torch.mean(g_error_tmp)

                G.zero_grad()
                D.zero_grad()
                g_error.backward()
                g_optimizer.step()

        if epoch % 1 == 0:
            print_metric(D, real_data, gen_data)
            print("\n\nD prob: ", round(torch.mean(D(real_data).data), 3))
            print("G prob: ", round(torch.mean(D(gen_data).data), 3))

        d_probs[epoch] = round(torch.mean(D(real_data).data), 3)
        g_probs[epoch] = round(torch.mean(D(gen_data).data), 3)


# --- plot --- #
plt.plot(d_probs, 'r')
plt.plot(g_probs, 'b')
plt.show()

plot_gan_result(data, gen_data)
plt.show()



