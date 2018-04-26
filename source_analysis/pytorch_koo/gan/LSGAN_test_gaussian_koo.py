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
            nn.Linear(2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 2),
            nn.Tanh()
        )
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(2, 1),
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
    n_row = 2
    dim = 2
    data = torch.randn([n_row, dim])
    data = Variable(data, requires_grad=False).type(torch.FloatTensor)

    # --- generator & discriminator --- #
    g_input_size = 2
    G = generator()
    D = discriminator()


    lrG = 0.0002
    lrD = 0.0004
    beta1 = 0.5
    beta2 = 0.999

    # --- optimizer --- #
    d_optimizer = optim.Adam(D.parameters(), lr=lrG, betas=(beta1, beta2))
    g_optimizer = optim.Adam(G.parameters(), lr=lrD, betas=(beta1, beta2))

    # --- parameters --- #
    epochs, d_epochs, g_epochs = 1000, 1, 1
    n_row = data.size()[0]
    batch_size = 2
    total_batch = n_row // batch_size

    if torch.cuda.is_available():
        D.cuda()
        G.cuda()

    d_probs = [None for _ in range(epochs)]
    g_probs = [None for _ in range(epochs)]

    g_grads = []

    for epoch in tqdm(range(epochs)):
        # data = data[np.random.choice(n_row, size=n_row, replace=False), :]  # shuffle

        for d_epoch in range(d_epochs):
            for batch_idx in range(total_batch):
                # make generated data
                # z_noise = torch.randn([batch_size, dim])

                z_noise = Variable(torch.Tensor([[1.1, 2.2], [3.3, 4.4]]), requires_grad=False).type(torch.FloatTensor).cuda()
                gen_data = G(z_noise)
                real_data = data[(batch_size * batch_idx):(batch_size * (batch_idx+1)), :]
                real_data = real_data.cuda()

                d_optimizer.zero_grad()

                d_error_1 = torch.mean((1/2)*((D(real_data)-1)**2))
                d_error_2 = torch.mean((1/2)*((D(gen_data)-0)**2))
                d_error = d_error_1 + d_error_2
                d_error.backward()
                d_optimizer.step()

        for g_epoch in range(g_epochs):
            for batch_idx in range(total_batch):
                # make generated data
                # z_noise = torch.randn([batch_size, dim])
                z_noise = Variable(torch.Tensor([[1.1, 2.2], [3.3, 4.4]]), requires_grad=False).type(torch.FloatTensor).cuda()
                g_optimizer.zero_grad()

                gen_data = G(z_noise)
                g_error_tmp = (1/2)*((D(gen_data)-1)**2)
                # g_error_tmp = -torch.log(1 - D(gen_data))
                g_error = torch.mean(g_error_tmp)
                g_error.backward()
                g_optimizer.step()


        if (epoch + 1) % 100 == 0:
            print_metric(D, real_data, gen_data)
            print("D_1_loss: %.4f ||| D_2_loss: %.4f ||| D_loss: %.4f ||| G_loss: %.4f ||| D_prob: %.4f ||| G_prob: %.4f  "
                  % (round(d_error_1.data[0], 3)
                  , round(d_error_2.data[0], 3)
                  , round(d_error.data[0], 3)
                  , round(g_error.data[0], 3)
                  , round(torch.mean(D(real_data).data), 3)
                  , round(torch.mean(D(gen_data).data), 3))
                  )
            plot_gan_result(data, gen_data)
            plt.show()

        d_probs[epoch] = round(torch.mean(D(real_data).data), 3)
        g_probs[epoch] = round(torch.mean(D(gen_data).data), 3)


# --- plot --- #
plt.plot(d_probs, 'r')
plt.plot(g_probs, 'b')
plt.show()



