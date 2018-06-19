

import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import random
import math
import os
import matplotlib.pyplot as plt
import copy
import preprocessing_LSTM as pre     # todo: 여기 밑줄 왜 뜨지...?
from tqdm import tqdm
import copy



# todo: GPU 세팅
# if torch.cuda.is_available():
#     x = x.cuda()

# todo - Question: forget gate 랑 output gate 랑 하는 일이 중복되는 것 아닌가 ㅡ
# todo: 이걸 고려한 것이 GRU 인가 ...?

# todo: layer 를 여러개 쌓는 것은 lstm 에서 어떻게 하지 ...?

# todo: bidirectional lstm 구현 ...?

# todo: many-to-one (1개, 여러개), many-to-many

# todo: rnn & gru 구현


if __name__ == "__main__":

    # --- directory --- #
    pre.get_dir()
    path = r"D:\\"
    pre.set_dir(path)

    # --- data --- #
    # 1) simple data

    # data_x = [x for x in range(10)]
    # data_x = pre.normalize(data_x, -0.5, 0.5)

    # 2) PQMS data

    data_x = pd.read_csv("./booksihwa_power_phase3.csv")
    data_x = data_x.ix[:, ["Time", "Load"]]
    data_x.set_index('Time', drop=True, inplace=True)
    data_x.index = pd.to_datetime(data_x.index)

    data_x = data_x.resample('1H', how={'Load': np.mean})  # todo: 이상치 제거하고 ㅡ 평균내야함
    data_x = data_x["Load"].tolist()

    data_x, max_data, min_data = pre.normalize(data_x, -0.5, 0.5)
    data_x_normalized, max_data, min_data = pre.normalize(data_x, -0.5, 0.5)
    data_x_denormalized = pre.de_normalize(data_x_normalized, max_data, min_data, max_set=0.5, min_set=-0.5)

    pre.plot('1', data_x)
    pre.plot('2', data_x_denormalized)
    pre.plot('3', data_x_normalized)
    pre.head(data_x, 10)

    # 3) sine data

    # data_x = np.sin(np.arange(0, 30, 0.1)).tolist()
    # data_x, max_data, min_data = pre.normalize(data_x, -0.5, 0.5)
    data_ori = copy.copy(data_x)

    # pre.plot(data_x)

    # --- preprocess data --- #
    n_col = 1
    n_row_input = 70
    n_row_output = 1
    n_row = pre.get_n_row(data_x)

    # make None value on 'data_x'
    # rand_idx = np.random.choice(np.arange(0, n_row, 1), size=n_row//10, replace=False)

    # todo: 처음 데이터에 nan 생기는 경우 제외 ㅡ 나중에 처음 데이터에 nan 있는 경우도 고려 필요
    rand_idx = np.random.choice(np.arange(1, n_row, 1), size=n_row // 10, replace=False)
    for idx in rand_idx:
        data_x[idx] = np.nan

    data_for_rnn_x, data_for_rnn_y = pre.get_list_type_data(data_x, n_row_input, n_row_output)
    data_ori_x, _ = pre.get_list_type_data(data_ori, n_row_input, n_row_output)

    first_data_nan_idxs = []
    for idx in range(len(data_for_rnn_x)):
        if pre.isnan(data_for_rnn_x[idx][0]):
            first_data_nan_idxs.append(idx)

    data_for_rnn_ori_x_without_first_data_nan = []
    data_for_rnn_x_without_first_data_nan = []
    data_for_rnn_y_without_first_data_nan = []
    for idx in range(len(data_for_rnn_x)):
        if idx not in first_data_nan_idxs:
            data_for_rnn_ori_x_without_first_data_nan.append(data_ori_x[idx])
            data_for_rnn_x_without_first_data_nan.append(data_for_rnn_x[idx])
            data_for_rnn_y_without_first_data_nan.append(data_for_rnn_y[idx])
    # len(data_for_rnn_x_without_first_data_nan)
    data_ori_x = data_for_rnn_ori_x_without_first_data_nan
    data_for_rnn_x = data_for_rnn_x_without_first_data_nan
    data_for_rnn_y = data_for_rnn_y_without_first_data_nan

    data_for_rnn_x[68]
    # data_for_rnn_y[0]
    # --- train setting --- #
    # todo: pre 로 받아온 함수들 정리 ㅡ preprocessing 과 lstm 관련쪽 나눔
    param = pre.Param()
    # optimizer = torch.optim.SGD(pre.parameters(param), lr=0.01)

    learning_rate = 0.1
    optimizer = torch.optim.Adam(pre.parameters(param), lr=learning_rate)
    num_iter = 1
    n = len(data_for_rnn_x)

    # --- train --- #
    # todo: 학습 잘 안됨...! 왜 그럴까...?

    # todo: loss surface 확인
    nan_idxs = []
    loss_arr = []

    iter_ = 0   # todo: 지움
    idx = 0     # todo: 지움

    for iter_ in tqdm(range(num_iter)):

        if iter_ >= 1:
            if np.isnan(loss.data[0]):
                nan_idxs.append([iter_, idx])
                print("Break...!")
                break

        # Learning rate decay
        if (iter_+1) % 5 == 0:
            learning_rate *= 0.01
            optimizer = torch.optim.Adam(pre.parameters(param), lr=learning_rate)
            print("Learning Rate:", learning_rate)

        cnt = 0
        for idx in range(n):
            # print("idx:", idx)
            cnt += 1
            data = data_for_rnn_x[idx]
            # y = data_for_rnn_y[idx]

            output, outputs = pre.lstm_forward_propagation_with_missing_data(data, param, verbose=False)

            # if idx is 67:
            #     output, outputs = pre.lstm_forward_propagation_with_missing_data(data, param, verbose=False)

            # todo: 왜 print 가 1회만 되지...?
            # print(outputs, "--- idx:", idx, "---iter_:", iter_)
            # todo: 데이터 몇개 비우고 이걸로 해봄 !!!!!!!!!!!!!!!!!!!!!!!!!!!
            # todo: 여기에 D 도 추가 !!!!!!!!!!!!!!!!!!!
            # todo: 이상한 논리 제거 !!!!!!!!!!!!!!!!!!!!

            # output, outputs = pre.lstm_forward_propagation(data, param, verbose=False)
            # todo: 학습이 잘 안되는 이유는.. t 시점의 값을 예측하는 것과, t+1 시점을 예측하는 것의 차이 아닐까 ...?
            # todo: 이 부분을 고려해서 ㅡ 구조를 변경해야 할 듯
            # todo: 이게 진짜 문제인지 ㅡ 맞는지 틀린지 정확히 모르겠음 ㅡ 생각 필요

            # loss = pre.loss_fn(y=y, pred=output)

            losses = []
            for idx_l in range(len(data)-1):
                if not pre.isnan(data[idx_l]):
                    losses.append(pre.loss_fn(data[idx_l], outputs[idx_l]))

            loss = sum(losses)

            param = pre.zero_grad(param)  # set gradients as 0
            if idx == 0:
                loss.backward(retain_graph=True)  # compute gradient of the loss
            loss.backward()
            optimizer.step()    # apply back propagation

            loss_val = round(loss.data[0], 3)

            if cnt % 100 == 0:
                print(loss_val)   # todo: 학습 잘 되다가 왜 nan 이 뜰까 ...? SGD 로 학습할 경우
            loss_arr.append(loss_val)

            if np.isnan(loss_val):
                nan_idxs.append([iter_, idx])
                print("Break...!")
                break

    pre.plot('4', loss_arr)

    pre.plot('5', data_x)





    data_range = np.max(data_ori) - np.min(data_ori)
    for idx in range(len(data_for_rnn_x)):
        data_ori_part = data_ori_x[idx]
        data = data_for_rnn_x[idx]
        y = data_for_rnn_y[idx]
        _, preds = pre.lstm_forward_propagation_with_missing_data(data, param, verbose=False)
        preds = [x.data[0] for x in preds]
        preds_variable = Variable(torch.DoubleTensor(preds), requires_grad=False)

        RMSE = torch.sum((preds_variable - data_ori_part).pow(2).sqrt()) / len(data_ori_part)
        NRMSE = RMSE / data_range

        print('RMSE:', RMSE.data[0], ', NRMSE:', NRMSE.data[0])


        plt.figure(figsize=(8,4))

        plt.title('IDX:' + str(idx) + ', RMSE:' + str(RMSE.data[0]) + ', NRMSE:' + str(NRMSE.data[0]))
        plt.plot(preds, color="black")
        plt.plot(list(data.data), color="xkcd:azure")
        plt.grid(True)

        plt.savefig('C:\\_data\\lstm_sine_plot\\' + str(idx) + '.png', bbox_inches='tight')

        plt.show()

    # --- param 변화 확인 --- #
    pre.print_param(param)
    pre.print_grad(param)







