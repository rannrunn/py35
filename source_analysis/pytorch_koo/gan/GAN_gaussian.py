# 작성자 : 구진혁

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
np.set_printoptions(threshold=np.nan)
import time

# 제너레이터 모델
def generator():
    return nn.Sequential(
        nn.Linear(2, 32),
        nn.BatchNorm1d(32),
        nn.LeakyReLU(0.2),
        nn.Linear(32, 2)
    )

# 디스크리미네이터 모델
def discriminator():
    return nn.Sequential(
        nn.Linear(2, 32),
        nn.BatchNorm1d(32),
        nn.LeakyReLU(0.2),
        nn.Linear(32, 32),
        nn.BatchNorm1d(32),
        nn.LeakyReLU(0.2),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

# 제너레이터와 디스크리미네이터 생성
G = generator().cuda()
D = discriminator().cuda()

# 파라미터
epochs = 2000
data_size = 3000;
mini_batch_size = 300;
data_dim = 2;

# 이미지 출력 에폭
epoch_image = 10

# 옵티마이저
lrD = 0.0005
lrG = 0.0005
beta1 = 0.5
beta2 = 0.999
g_optimizer = torch.optim.Adam(G.parameters(), lr=lrG, betas=(beta1, beta2))
d_optimizer = torch.optim.Adam(D.parameters(), lr=lrG, betas=(beta1, beta2))

# 손실 계산 시 사용되는 인자 : 1, 0
y_real_, y_fake_ = Variable(torch.ones(mini_batch_size, 1).cuda()), Variable(torch.zeros(mini_batch_size, 1).cuda())

# 손실함수 : Binary Cross Entory
bceloss = nn.BCELoss().cuda()
# 손실함수 : Mean Squared Error
mseloss = nn.MSELoss().cuda()

# 실제 데이터
data = Variable(torch.randn([data_size, data_dim])).cuda()

# scatter OR contour 에 사용할 변수
axis_length = 7
axis_min = -1 * axis_length
axis_max = axis_length + 0.1
matrix_scale = (axis_length * 20) + 1

ndarray_2dim = np.mgrid[axis_min:axis_max:0.1, axis_min:axis_max:0.1].reshape(2, -1).T
ndarray_1dim = np.arange(axis_min, axis_max, 0.1)

contour_line_distance_value = 3

# 학습
for epoch in range(epochs):

    # 실제 데이터를 랜덤하게 재배치
    data = data[np.random.choice(data_size, size=data_size, replace=False), :]

    # 제너레이터가 한 에폭 당 생성하는 데이터를 저장하기 위한 torch variable 변수 생성
    g_epoch_data = Variable(torch.FloatTensor()).cuda()

    # 디스크리미네이터의 학습 전 상태를 저장하기 위한 torch variable 변수 생성
    d_before = Variable(torch.rand([matrix_scale, matrix_scale])).cuda()

    # 디스크리미네이터의 학습 후 상태를 저장하기 위한 torch variable 변수 생성
    d_after = Variable(torch.rand([matrix_scale, matrix_scale])).cuda()

    # 디스크리미네이터
    for i in range(0, data_size, mini_batch_size):

        # 디스크리미네이터의 학습 전 상태
        d_before = D(Variable(torch.FloatTensor(ndarray_2dim)).cuda()).cpu().data.numpy().reshape(matrix_scale, matrix_scale)

        # 배치 학습 시 사용되는 실제 데이터를 가져옴
        r_batch_data = data[i:i + mini_batch_size]
        # 배치 학습 시 사용되는 랜덤 데이터 생성
        z_ = Variable(torch.randn([mini_batch_size, data_dim])).cuda()

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
        d_loss.backward(retain_graph=True)
        d_optimizer.step()

        d_after = D(Variable(torch.FloatTensor(ndarray_2dim)).cuda()).cpu().data.numpy().reshape(matrix_scale, matrix_scale)
        d_plt_bool = False
        if d_plt_bool:

            # 학습 전
            plt.figure(figsize=(6, 6))
            plt.contourf(ndarray_1dim, ndarray_1dim, d_before, alpha=.75, cmap='jet')
            C = plt.contour(ndarray_1dim, ndarray_1dim, d_before, colors='black', linewidth=0.5)
            plt.clabel(C, inline=1, fontsize=10)
            plt.scatter(r_batch_data.cpu().data.numpy()[:, :1], r_batch_data.cpu().data.numpy()[:, 1:2], color='yellow', marker= '*', label='RealBatchData')
            plt.scatter(g_batch_data.cpu().data.numpy()[:, :1], g_batch_data.cpu().data.numpy()[:, 1:2], color='red', marker='o', label='GenBatchData')
            plt.scatter(z_.cpu().data.numpy()[:, :1], z_.cpu().data.numpy()[:, 1:2], color='purple', marker='+', label='Z_BatchData')
            plt.xlim(axis_min, axis_max)
            plt.ylim(axis_min, axis_max)
            plt.legend()
            plt.title('Discriminator(Before) Epoch: [%d/%d], Batch: [%d/%d]\nDRealLoss: %.4f, DFakeLoss: %.4f'
                      % (epoch + 1, epochs, (i // mini_batch_size) + 1, data_size // mini_batch_size, d_real_loss.data[0], d_fake_loss.data[0]))
            plt.show()
            plt.close()

            # 학습 후
            plt.figure(figsize=(6, 6))
            plt.contourf(ndarray_1dim, ndarray_1dim, d_after, alpha=.75, cmap='jet')
            C = plt.contour(ndarray_1dim, ndarray_1dim, d_after, colors='black', linewidth=0.5)
            plt.clabel(C, inline=1, fontsize=10)
            plt.scatter(r_batch_data.cpu().data.numpy()[:, :1], r_batch_data.cpu().data.numpy()[:, 1:2], color='yellow', marker= '*', label='RealBatchData')
            plt.scatter(g_batch_data.cpu().data.numpy()[:, :1], g_batch_data.cpu().data.numpy()[:, 1:2], color='red', marker='o', label='GenBatchData')
            plt.scatter(z_.cpu().data.numpy()[:, :1], z_.cpu().data.numpy()[:, 1:2], color='purple', marker='+', label='Z_BatchData')
            plt.xlim(axis_min, axis_max)
            plt.ylim(axis_min, axis_max)
            plt.legend()
            plt.title('Discriminator(After) Epoch: [%d/%d], Batch: [%d/%d]\nDRealLoss: %.4f, DFakeLoss: %.4f'
                      % (epoch + 1, epochs, (i // mini_batch_size) + 1, data_size // mini_batch_size, d_real_loss.data[0], d_fake_loss.data[0]))
            plt.show()
            plt.close()


    # 제너레이터
    for i in range(0, data_size, mini_batch_size):

        # 배치할습 시 사용되는 랜덤 데이터 생성
        z_ = Variable(torch.randn([mini_batch_size, data_dim])).cuda()

        # 1. 제너레이터가 학습할 가짜 데이터를 생성
        # 2. 기울기를 초기화
        # 3. 제너레이터의 손실을 계산
        # 4~5. 옵티마이저를 사용해 백프로파게이션 진행
        g_batch_data = G(z_)
        g_optimizer.zero_grad()
        g_loss = bceloss(D(g_batch_data), y_real_)
        g_loss.backward(retain_graph=True)
        g_optimizer.step()

        # 제너레이터가 한 에폭 동안 생성한 데이터를 누적하여 저장
        g_after_batch_data = G(z_)
        if len(list(g_epoch_data.size())) == 0:
            g_epoch_data = g_after_batch_data
        else:
            g_epoch_data = torch.cat([g_epoch_data, g_after_batch_data])


    g_plt_bool = True
    if g_plt_bool and (epoch + 1) % 10 == 0:

        print('Epoch: [%d/%d] DRealLoss: %.4f, DFakeLoss: %.4f, DLoss: %.4f, GLoss: %.4f'
                  % (epoch + 1, epochs, d_real_loss.data[0], d_fake_loss.data[0], d_loss.data[0], g_loss.data[0]))


        plt.scatter(data.cpu().data.numpy()[:, :1], data.cpu().data.numpy()[:, 1:2], color='yellow', marker= '*', label='RealData')
        plt.scatter(z_.cpu().data.numpy()[:, :1], z_.cpu().data.numpy()[:, 1:2], color='purple', marker='+', label='Z_Data')
        plt.scatter(g_epoch_data.cpu().data.numpy()[:, :1], g_epoch_data.cpu().data.numpy()[:, 1:2], color='green', marker='^', label='GenEpochData')
        plt.scatter(g_after_batch_data.cpu().data.numpy()[:, :1], g_after_batch_data.cpu().data.numpy()[:, 1:2], color='red', marker='^', label='GenAfterData')
        plt.xlim(axis_min, axis_max)
        plt.ylim(axis_min, axis_max)
        plt.legend()
        plt.title('Epoch: [%d/%d] DRealLoss: %.4f, DFakeLoss: %.4f, GLoss: %.4f'
                  % (epoch + 1, epochs, d_real_loss.data[0], d_fake_loss.data[0], g_loss.data[0]))
        plt.show()
        print('GEpochMean: %.4f, GEpochSTD: %.4f' % (g_epoch_data.mean(), g_epoch_data.std()))



        plt.figure(figsize=(6, 6))
        # 올바르게 행렬연산을 하였는지 체크
        # check_matrix = D(Variable(torch.FloatTensor([[-4, -4], [-4, 4], [0, 0], [4, -4], [4, 4]])).cuda())
        # print(check_matrix)
        plt.contourf(ndarray_1dim, ndarray_1dim, d_after, contour_line_distance_value, alpha=.75, cmap='jet')
        C = plt.contour(ndarray_1dim, ndarray_1dim, d_after, contour_line_distance_value, colors='black', linewidth=0.5)
        plt.clabel(C, inline=1, fontsize=10)
        plt.scatter(data.cpu().data.numpy()[:, :1], data.cpu().data.numpy()[:, 1:2], color='yellow', marker= '*', label='RealData')
        plt.scatter(z_.cpu().data.numpy()[:, :1], z_.cpu().data.numpy()[:, 1:2], color='purple', marker='+', label='Z_Data')
        plt.scatter(g_batch_data.cpu().data.numpy()[:, :1], g_batch_data.cpu().data.numpy()[:, 1:2], color='blue', marker='o', label='GenBeforeData')
        plt.scatter(g_after_batch_data.cpu().data.numpy()[:, :1], g_after_batch_data.cpu().data.numpy()[:, 1:2], color='red', marker='^', label='GenAfterData')
        plt.xlim(axis_min, axis_max)
        plt.ylim(axis_min, axis_max)
        plt.legend()
        plt.title('Generator(After) Epoch: [%d/%d], Batch: [%d/%d]\n DRealLoss: %.4f, DFakeLoss: %.4f, GLoss: %.4f'
                  % (epoch + 1, epochs, (i // mini_batch_size) + 1, data_size // mini_batch_size, d_real_loss.data[0], d_fake_loss.data[0], g_loss.data[0]))
        plt.show()
        plt.close()


