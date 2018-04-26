import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

epochs = 1000

def generator():
    return nn.Sequential(
        nn.Linear(5, 50),
        nn.LeakyReLU(0.2),
        nn.Linear(50, 200),
        nn.LeakyReLU(0.2),
        nn.Linear(200, 100),
        nn.LeakyReLU(0.2),
        nn.Linear(100, 50),
        nn.LeakyReLU(0.2),
        nn.Linear(50, 5),
        nn.Tanh()
    )

def discriminator():
    return nn.Sequential(
        nn.Linear(5, 50),
        #nn.BatchNorm1d(50),
        nn.LeakyReLU(0.2),
        nn.Linear(50, 100),
        #nn.BatchNorm1d(100),
        nn.LeakyReLU(0.2),
        nn.Linear(100, 50),
        #nn.BatchNorm1d(50),
        nn.LeakyReLU(0.2),
        nn.Linear(50, 1),
        nn.Sigmoid()
    )


G = generator().cuda()
D = discriminator().cuda()

batch_size = 10;
data_dim = 5;


lrD = 0.001
lrG = 0.001
beta1 = 0.5
beta2 = 0.999
g_optimizer = torch.optim.Adam(G.parameters(), lr=lrG, betas=(beta1, beta2))
d_optimizer = torch.optim.Adam(D.parameters(), lr=lrG, betas=(beta1, beta2))

y_real_, y_fake_ = Variable(torch.ones(batch_size, 1).cuda()), Variable(torch.zeros(batch_size, 1).cuda())

bceloss = nn.BCELoss()
mseloss = nn.MSELoss()


data = Variable(torch.rand(batch_size, data_dim)).cuda()
z_ = Variable(torch.rand(batch_size, data_dim)).cuda()
print('data:', data)
# print('data:', z_)

# real_tensor = torch.Tensor([0.6, 0.5, 0.8, 0.3 , 0.1, 0.2]);
# real = Variable(real_tensor, requires_grad=False).type(torch.FloatTensor).cuda()
# z_ = Variable(z_, requires_grad=False).type(torch.FloatTensor).cuda()

for epoch in range(epochs):
    # discriminator
    for i in range(20):

        d_optimizer.zero_grad()

        d_real_loss = bceloss(D(data), y_real_)
        d_fake_loss = bceloss(D(G(z_)), y_fake_)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward(retain_graph=True)

        d_optimizer.step()

    # generator
    for i in range (20):

        g_optimizer.zero_grad()
        g_loss = bceloss(D(G(z_)), y_real_)
        g_loss.backward(retain_graph=True)

        g_optimizer.step()

    if (epoch + 1) % 10 == 0:
        print('Epoch: [%d/%d] DRealLoss: %.4f, DFakeLoss: %.4f, DLoss: %.4f, GLoss: %.4f'
              % (epoch + 1, epochs, d_real_loss.data[0], d_fake_loss.data[0], d_loss.data[0], g_loss.data[0]))
        plt.scatter(data.cpu().data.numpy()[:, :1], data.cpu().data.numpy()[:, 1:2], color='yellow', marker= '*')
        plt.scatter(G(z_).cpu().data.numpy()[:, :1], G(z_).cpu().data.numpy()[:, 1:2], color='green', marker= '^')
        #plt.scatter()
        plt.show()

