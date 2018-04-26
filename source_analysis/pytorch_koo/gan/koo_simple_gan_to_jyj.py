import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

def generator():
    return nn.Sequential(
        nn.Linear(2, 128),
        nn.LeakyReLU(0.2),
        nn.Linear(128, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 128),
        nn.LeakyReLU(0.2),
        nn.Linear(128, 64),
        nn.LeakyReLU(0.2),
        nn.Linear(64, 2)
    )

def discriminator():
    return nn.Sequential(
        nn.Linear(2, 16),
        #nn.BatchNorm1d(50),
        nn.LeakyReLU(0.2),
        nn.Linear(16, 32),
        #nn.BatchNorm1d(100),
        nn.LeakyReLU(0.2),
        nn.Linear(32, 64),
        #nn.BatchNorm1d(50),
        nn.LeakyReLU(0.2),
        nn.Linear(64, 32),
        nn.LeakyReLU(0.2),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )


G = generator().cuda()
D = discriminator().cuda()


epochs = 1000
data_size = 3000;
batch_size = 300;
data_dim = 2;


lrD = 0.001
lrG = 0.001
beta1 = 0.5
beta2 = 0.999
g_optimizer = torch.optim.Adam(G.parameters(), lr=lrG, betas=(beta1, beta2))
d_optimizer = torch.optim.Adam(D.parameters(), lr=lrG, betas=(beta1, beta2))

y_real_, y_fake_ = Variable(torch.ones(batch_size, 1).cuda()), Variable(torch.zeros(batch_size, 1).cuda())

bceloss = nn.BCELoss()
mseloss = nn.MSELoss()


data = Variable(torch.randn([data_size, data_dim])).cuda()

print('data:', data)

# real_tensor = torch.Tensor([0.6, 0.5, 0.8, 0.3 , 0.1, 0.2]);
# real = Variable(real_tensor, requires_grad=False).type(torch.FloatTensor).cuda()
# z_ = Variable(z_, requires_grad=False).type(torch.FloatTensor).cuda()

for epoch in range(epochs):

    data = data[np.random.choice(data_size, size=data_size, replace=False), :]

    g_vars = Variable(torch.FloatTensor()).cuda()

    # discriminator
    for i in range(0, data_size, batch_size):

        batch_data = data[i:i+batch_size]
        z_ = Variable(torch.randn([batch_size, data_dim])).cuda()

        generation_data = G(z_)
        d_optimizer.zero_grad()
        d_real_loss = mseloss(D(batch_data), y_real_)
        d_fake_loss = mseloss(D(generation_data), y_fake_)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward(retain_graph=True)

        d_optimizer.step()

    # generator
    for i in range(0, data_size, batch_size):

        z_ = Variable(torch.randn([batch_size, data_dim])).cuda()

        generation_data = G(z_)
        g_optimizer.zero_grad()
        g_loss = mseloss(D(generation_data), y_real_)
        g_loss.backward(retain_graph=True)

        g_optimizer.step()

        if len(list(g_vars.size())) == 0:
            g_vars = generation_data
        else:
            g_vars = torch.cat([g_vars, generation_data])


    if (epoch + 1) % 20 == 0:
        print('Epoch: [%d/%d] DRealLoss: %.4f, DFakeLoss: %.4f, DLoss: %.4f, GLoss: %.4f'
              % (epoch + 1, epochs, d_real_loss.data[0], d_fake_loss.data[0], d_loss.data[0], g_loss.data[0]))
        plt.scatter(data.cpu().data.numpy()[:, :1], data.cpu().data.numpy()[:, 1:2], color='yellow', marker= '*')
        plt.scatter(g_vars.cpu().data.numpy()[:, :1], g_vars.cpu().data.numpy()[:, 1:2], color='green', marker='^')
        plt.show()

