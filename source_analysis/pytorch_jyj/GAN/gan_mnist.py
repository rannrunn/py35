# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/generative_adversarial_network/main.py

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable

import matplotlib.pyplot as plt

import pandas as pd
import os


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# todo: denorm 이 뭐하는 것일까...?
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# todo: tranforms.Compose 가 뭐하는 것일까...?
# Image processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))])

# MNIST dataset
mnist = datasets.MNIST(root='./data/',
                       train=True,
                       transform=transform,
                       download=True)
# Data loader
# todo: DataLoader 는 어떻게 사용하는 것일까...?
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=100,
                                          shuffle=True)

if __name__ == "__main__":
    # Discriminator
    D = nn.Sequential(
        nn.Linear(784, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid())

    # Generator
    G = nn.Sequential(
        nn.Linear(64, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 784),
        nn.Tanh())

    if torch.cuda.is_available():
        D.cuda()
        G.cuda()

    # Binary cross entropy loss and optimizer
    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

    epochs = 500
    n_row = 600
    d_losses_real = [None for _ in range(epochs * n_row)]
    d_losses_fake = [None for _ in range(epochs * n_row)]
    g_losses = [None for _ in range(epochs * n_row)]
    d_probs = [None for _ in range(epochs * n_row)]
    g_probs = [None for _ in range(epochs * n_row)]

    cnt = -1    # cnt: count

    # Start training
    for epoch in range(epochs):
        for i, (images, _) in enumerate(data_loader):
            cnt += 1
            # Build mini-batch dataset
            batch_size = images.size(0)
            images = to_var(images.view(batch_size, -1))

            # Create the labels which are later used as input for the BCE loss
            real_labels = to_var(torch.ones(batch_size))
            fake_labels = to_var(torch.zeros(batch_size))

            # ============= Train the discriminator =============#
            # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
            # Second term of the loss is always zero since real_labels == 1
            outputs = D(images)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs

            # Compute BCELoss using fake images
            # First term of the loss is always zero since fake_labels == 0
            z = to_var(torch.randn(batch_size, 64))
            fake_images = G(z)
            outputs = D(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            # records errors
            d_losses_real[cnt] = d_loss_real
            d_losses_fake[cnt] = d_loss_fake
            d_probs[cnt] = real_score.data.mean()
            g_probs[cnt] = fake_score.data.mean()

            # Backprop + Optimize
            d_loss = d_loss_real + d_loss_fake
            D.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # =============== Train the generator ===============#
            # Compute loss with fake images
            z = to_var(torch.randn(batch_size, 64))
            fake_images = G(z)
            outputs = D(fake_images)

            # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
            # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
            g_loss = criterion(outputs, real_labels)

            # records errors
            g_losses[cnt] = g_loss

            # Backprop + Optimize
            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if (i + 1) % 300 == 0:
                print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, '
                      'g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f'
                      % (epoch, epochs, i + 1, 600, d_loss.data[0], g_loss.data[0],
                         real_score.data.mean(), fake_score.data.mean()))

        # Save real images
        if (epoch + 1) == 1:
            images = images.view(images.size(0), 1, 28, 28)
            save_image(denorm(images.data), './data/real_images.png')

        # Save sampled images
        fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
        save_image(denorm(fake_images.data), './data/fake_images-%d.png' % (epoch + 1))

    # Save the trained parameters
    # todo: trained parameter 저장하고 불러오는 tutorial code 생성
    torch.save(G.state_dict(), './generator.pkl')
    torch.save(D.state_dict(), './discriminator.pkl')
    # # Save and load the entire model.
    # https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
    # torch.save(resnet, 'model.pkl')
    # model = torch.load('model.pkl')
    #
    # # Save and load only the model parameters(recommended).
    # torch.save(resnet.state_dict(), 'params.pkl')
    # resnet.load_state_dict(torch.load('params.pkl'))

    d_losses_real = list(map(lambda x: x.data[0], d_losses_real))
    d_losses_fake = list(map(lambda x: x.data[0], d_losses_fake))
    g_losses = list(map(lambda x: x.data[0], g_losses))
    d_probs
    g_probs

    plt.plot(d_losses_real)
    plt.show()
    plt.close()

    os.getcwd()
    data = pd.read_csv("exp_res.csv")
    data.columns

    plt.plot(data['d_losses_fake'], 'lightcoral')
    plt.plot(data['d_losses_real'], 'r')
    plt.show()
    plt.close()
