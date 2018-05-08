# 작성자 : 구진혁

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
np.set_printoptions(threshold=np.nan)
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import scipy.misc
import time


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

# 파라미터
epochs = 2000
mini_batch_size = 400;
g_input_size = 64;

train_dataset = dsets.MNIST(
    root='./data/',
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))]),
    download=True
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=mini_batch_size,
    shuffle=True
)

# 제너레이터 모델
def generator():
    return nn.Sequential(
        nn.Linear(g_input_size, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, 784),
        nn.Tanh(),
    )

# 디스크리미네이터 모델
def discriminator():
    return nn.Sequential(
        nn.Linear(784, 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

# 제너레이터와 디스크리미네이터 생성
G = generator().cuda()
D = discriminator().cuda()


# 옵티마이저
lrD = 0.0005
lrG = 0.0005
beta1 = 0.5
beta2 = 0.999
d_optimizer = torch.optim.Adam(D.parameters(), lr=lrD, betas=(beta1, beta2))
g_optimizer = torch.optim.Adam(G.parameters(), lr=lrG, betas=(beta1, beta2))


# 손실 계산 시 사용되는 인자 : 1, 0
y_real_, y_fake_ = Variable(torch.ones(mini_batch_size, 1).cuda()), Variable(torch.zeros(mini_batch_size, 1).cuda())

# 손실함수 : Binary Cross Entory
bceloss = nn.BCELoss().cuda()
# 손실함수 : Mean Squared Error
mseloss = nn.MSELoss().cuda()

# scatter OR contour 에 사용할 변수
axis_length = 9
axis_min = -1 * axis_length
axis_max = axis_length + 0.1

# 학습
for epoch in range(epochs):

    # 디스크리미네이터
    for i, (images, _) in enumerate(train_loader):
        # 배치 학습 시 사용되는 실제 데이터를 가져옴
        r_batch_data = Variable(images.view(mini_batch_size, 784)).cuda()

        # 배치 학습 시 사용되는 랜덤 데이터 생성
        z_ = Variable(torch.randn([mini_batch_size, g_input_size])).cuda()

        # 1. 디스크리미네이터가 학습할 가짜 데이터를 제너레이터를 통해 생성
        # 2. 기울기를 초기화
        # 3. 디스크리미네이터의 손실을 계산하기 위해 실제데이터에 대한 손실을 계산
        # 4. 디스크리미네이터의 손실을 계산하기 위해 가짜데이터에 대한 손실을 계산
        # 5. 실제데이터에 대한 손실과 가짜데이터에 대한 손실을 합함
        # 6~7. 옵티마이저를 사용해 백프로파게이션 진행
        g_batch_data = G(z_)
        d_optimizer.zero_grad()
        d_real_loss = bceloss(D(r_batch_data), y_real_)
        d_fake_loss = bceloss(D(g_batch_data), y_fake_)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        # 2. 기울기를 초기화
        # 3. 제너레이터의 손실을 계산
        # 4~5. 옵티마이저를 사용해 백프로파게이션 진행
        g_batch_data = G(z_)
        g_optimizer.zero_grad()
        g_loss = bceloss(D(g_batch_data), y_real_)
        g_loss.backward()
        g_optimizer.step()

    g_plt_bool = True
    if g_plt_bool and (epoch + 1) % 1 == 0:

        print('Epoch: [%d/%d] DRealLoss: %.4f, DFakeLoss: %.4f, DLoss: %.4f, GLoss: %.4f'
                  % (epoch + 1, epochs, d_real_loss.data[0], d_fake_loss.data[0], d_loss.data[0], g_loss.data[0]))
        # r_data_image = r_batch_data.cpu().data.numpy().reshape(mini_batch_size, 28, 28, 1)
        # r_images = np.squeeze(merge(r_data_image, [20, 20]))
        # scipy.misc.imsave('./mnist_real_' + str(epoch + 1) + '.png', r_images)
        g_data_image = g_batch_data.cpu().data.numpy().reshape(mini_batch_size, 28, 28, 1)
        g_images = np.squeeze(merge(g_data_image, [20, 20]))
        scipy.misc.imsave('./mnist_fake_' + str(epoch + 1) + '.png', g_images)

        print('d_max:', r_batch_data.data[0].max())
        print('d_min:', r_batch_data.data[0].min())
        print('g_max:', g_batch_data.data[0].max())
        print('g_min:', g_batch_data.data[0].min())

