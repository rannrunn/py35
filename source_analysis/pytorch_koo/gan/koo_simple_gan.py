import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable

epochs = 1000

def generator():
    return nn.Sequential(
        nn.Linear(1, 10),
        nn.Linear(10, 10),
        nn.Linear(10, 1),
        nn.Tanh()
    )

def discriminator():
    return nn.Sequential(
        nn.Linear(1, 10),
        nn.Linear(10, 10),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )

loss = nn.BCELoss()

real = Variable(torch.Tensor([0.8]), requires_grad=False).type(torch.FloatTensor).cuda()
z_ = Variable(torch.Tensor([-0.5]), requires_grad=False).type(torch.FloatTensor).cuda()

G = generator().cuda()
D = discriminator().cuda()

print(G(z_))

lrD = 0.0001
lrG = 0.0001
beta1 = 0.5
beta2 = 0.999
g_optimizer = torch.optim.Adam(G.parameters(), lr=lrG, betas=(beta1, beta2))
d_optimizer = torch.optim.Adam(D.parameters(), lr=lrG, betas=(beta1, beta2))

y_real_, y_fake_ = Variable(torch.ones(1, 1).cuda()), Variable(torch.zeros(1, 1).cuda())

bceloss = nn.BCELoss()

for epoch in range(epochs):
    # discriminator
    for i in range(10):

        d_optimizer.zero_grad()

        d_real_loss = bceloss(D(real), y_real_)
        d_fake_loss = bceloss(D(G(z_)), y_fake_)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward(retain_graph=True)

        d_optimizer.step()


    # generator
    for i in range (10):

        g_optimizer.zero_grad()

        g_loss = bceloss(D(G(z_)), y_real_)
        g_loss.backward(retain_graph=True)

        g_optimizer.step()

    if (epoch + 1) % 10 == 0:
        print('Epoch: [%d/%d] DRealLoss: %.4f, DFakeLoss: %.4f, DLoss: %.4f, GLoss: %.4f, real: %.4f, fake: %.4f'
              % (epoch, epochs, d_real_loss.data[0], d_fake_loss.data[0], d_loss.data[0], g_loss.data[0], real, G(z_)))


