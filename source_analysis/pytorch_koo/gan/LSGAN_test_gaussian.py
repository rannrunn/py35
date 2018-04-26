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


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 2048),
            nn.Tanh()
        )
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = out.view(-1, 1024, 2)
        return out


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


def print_metric(d_model, real_data_, gen_data_):
    print("\n\nD prob (on real data): ", round(torch.mean(d_model(real_data_).data), 3))
    print("D prob (on gen data): ", round(torch.mean(d_model(gen_data_).data), 3))


if __name__ == "__main__":
    # --- data --- #
    # 1) normal: unimodal
    dim_1 = 1024
    dim_2 = 2
    g_input_dim = 64
    epochs, d_epochs, g_epochs = 1000, 1, 1
    batch_size = 2
    total_batch = 1
    data = torch.randn(batch_size, dim_1, dim_2)
    data = Variable(data, requires_grad=False).type(torch.FloatTensor)

    # --- generator & discriminator --- #
    G = generator()
    D = discriminator()

    if torch.cuda.is_available():
        D.cuda()
        G.cuda()

    # --- optimizer --- #
    learning_rate = 0.0003
    d_optimizer = optim.Adam(D.parameters(), lr=learning_rate)
    g_optimizer = optim.Adam(G.parameters(), lr=learning_rate)

    # --- parameters --- #

    print('total_batch', total_batch)

    d_probs = [None for _ in range(epochs)]
    g_probs = [None for _ in range(epochs)]

    for epoch in tqdm(range(epochs)):
        # x_data = data[np.random.choice(n_row, size=n_row, replace=False), :]  # shuffle
        x_data = data

        # for item in xdata:
        #     print(item)

        for d_epoch in range(d_epochs):
            for batch_idx in range(total_batch):
                # make generated data
                z_noise = torch.randn([batch_size, g_input_dim])
                z_noise = Variable(z_noise, requires_grad=False).type(torch.FloatTensor).cuda()
                gen_data = G(z_noise)
                real_data = x_data[(batch_size * batch_idx):(batch_size * (batch_idx + 1)), :]
                real_data = real_data.cuda()


                d_error_1 = (1/2)*((D(real_data)-1)**2)
                d_error_2 = (1/2)*((D(gen_data)-0)**2)
                d_error_1_mean = torch.mean(d_error_1)
                d_error_2_mean = torch.mean(d_error_2)
                d_error =  d_error_1 + d_error_2
                d_error = torch.mean(d_error)
                D.zero_grad()
                d_error.backward()
                d_optimizer.step()

        for g_epoch in range(g_epochs):
            for batch_idx in range(total_batch):
                # make generated data
                z_noise = torch.randn([batch_size, g_input_dim])
                z_noise = Variable(z_noise, requires_grad=False).type(torch.FloatTensor).cuda()
                gen_data = G(z_noise)

                g_error = (1/2)*((D(gen_data)-1)**2)

                g_error = torch.mean(g_error)

                G.zero_grad()
                D.zero_grad()
                g_error.backward()
                g_optimizer.step()

        if (epoch + 1) % 100 == 0:
            print_metric(D, real_data, gen_data)
            print("\n\nD prob: ", round(torch.mean(D(real_data).data), 3))
            print("G prob: ", round(torch.mean(D(gen_data).data), 3))
            print('d_error_1_mean:', d_error_1_mean.data[0])
            print('d_error_2_mean:', d_error_2_mean.data[0])
            print('d_error_mean:', d_error.data[0])
            print('g_error:', g_error.data[0])
            # plot_gan_result(data, gen_data)
            plt.plot(np.asarray(x_data.data)[0], np.asarray(x_data.data)[1], '*')
            plt.plot(np.asarray(gen_data.data)[0], np.asarray(gen_data.data)[1], '^')
            plt.show()
        print('mean', torch.mean(D(real_data).data))
        d_probs[epoch] = round(torch.mean(D(real_data).data), 3)
        g_probs[epoch] = round(torch.mean(D(gen_data).data), 3)


# --- plot --- #
plt.plot(d_probs, 'r')
plt.plot(g_probs, 'b')
plt.show()



