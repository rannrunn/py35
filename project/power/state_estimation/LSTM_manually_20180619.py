import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import random
import math
import os
import matplotlib.pyplot as plt
import copy
import preprocessing_LSTM_20180619 as pre     # todo: 여기 밑줄 왜 뜨지...?
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
    path = r"D:\소프트팩토리\소프트팩토리_대전\data\PQMS"
    pre.set_dir(path)

    # --- data --- #
    # 1) simple data

    # data_x = [x for x in range(10)]
    # data_x = pre.normalize(data_x, -0.5, 0.5)

    # 2) PQMS data

    # data_x = pd.read_csv("buksiwha_pqms_phase_3.csv")
    # data_x = data_x.ix[:, ["Time", "Load"]]
    # data_x.set_index('Time', drop=True, inplace=True)
    # data_x.index = pd.to_datetime(data_x.index)
    #
    # data_x = data_x.resample('1H', how={'Load': np.mean})  # todo: 이상치 제거하고 ㅡ 평균내야함
    # data_x = data_x["Load"].tolist()
    #
    # data_x, max_data, min_data = pre.normalize(data_x, -0.5, 0.5)
    # data_x_normalized, max_data, min_data = pre.normalize(data_x, -0.5, 0.5)
    # data_x_denormalized = pre.de_normalize(data_x_normalized, max_data, min_data, max_set=0.5, min_set=-0.5)

    # pre.plot(data_x)
    # pre.plot(data_x_denormalized)
    # pre.plot(data_x_normalized)
    # pre.head(data_x, 10)

    # 3) sine data

    data_x = np.sin(np.arange(0, 30, 0.1)).tolist()
    data_x, max_data, min_data = pre.normalize(data_x, -0.5, 0.5)

    # pre.plot(data_x)

    # --- preprocess data --- #
    n_col = 1
    n_row_input = 70 * 2
    n_row_output = 1
    n_row = pre.get_n_row(data_x)

    # make None value on 'data_x'
    # rand_idx = np.random.choice(np.arange(0, n_row, 1), size=n_row//10, replace=False)

    # todo: 처음 데이터에 nan 생기는 경우 제외 ㅡ 나중에 처음 데이터에 nan 있는 경우도 고려 필요
    rand_idx = np.random.choice(np.arange(1, n_row, 1), size=n_row // 10, replace=False)
    for idx in rand_idx:
        data_x[idx] = np.nan

    data_for_rnn_x, _ = pre.get_list_type_data(data_x, n_row_input, n_row_output)

    first_data_nan_idxs = []
    for idx in range(len(data_for_rnn_x)):
        if pre.isnan(data_for_rnn_x[idx][0]):
            first_data_nan_idxs.append(idx)

    data_for_rnn_x_without_first_data_nan = []
    for idx in range(len(data_for_rnn_x)):
        if idx not in first_data_nan_idxs:
            data_for_rnn_x_without_first_data_nan.append(data_for_rnn_x[idx])
    # len(data_for_rnn_x_without_first_data_nan)
    data_for_rnn_x = data_for_rnn_x_without_first_data_nan
    len(data_for_rnn_x)
    # --------------------------------------------------------------------- #

    # data_for_rnn_y[0]
    # --- train setting --- #
    # todo: pre 로 받아온 함수들 정리 ㅡ preprocessing 과 lstm 관련쪽 나눔
    param = pre.Param()
    # optimizer = torch.optim.SGD(pre.parameters(param), lr=0.01)

    learning_rate = 0.1
    optimizer = torch.optim.Adam(pre.parameters(param), lr=learning_rate)
    num_iter = 15
    n = len(data_for_rnn_x)

    # --- train --- #
    # todo: 학습 잘 안됨...! 왜 그럴까...?

    # todo: loss surface 확인
    nan_idxs = []
    loss_arr = []

    iter_ = 0   # todo: 지움
    idx = 0     # todo: 지움

    for iter_ in tqdm(range(num_iter)):

        # Learning rate decay
        if iter_ % 5 == 0:
            learning_rate *= 1/2
            optimizer = torch.optim.Adam(pre.parameters(param), lr=learning_rate)
            print("Learning Rate:", learning_rate)

        cnt = 0
        for idx in range(n):
            # print("idx:", idx)
            cnt += 1
            data = data_for_rnn_x[idx]
            # y = data_for_rnn_y[idx]

            output, outputs, output_finals = pre.lstm_forward_propagation_with_missing_data(data, param, verbose=False)

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

            # == x(t+1) == #
            losses = []
            next_time = 1
            # todo: 처음에 hidden 이랑 cell 값 안 받는 애는 어떻게 학습시키지 ...?
            for idx_l in range(len(data)-next_time):
                if not pre.isnan(data[idx_l+next_time]):
                    losses.append(pre.loss_fn(data[idx_l + next_time], output_finals[idx_l]))

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

pre.plot(loss_arr)

pre.plot(data_x)

pre.plot([x for x in data.data])

idx = 0

idx += 10
data = data_for_rnn_x[idx]
# y = data_for_rnn_y[idx]
_, preds, pred_finals = pre.lstm_forward_propagation_with_missing_data(data, param, verbose=False)
preds = [x.data[0] for x in preds]
pred_finals = [x.data[0] for x in pred_finals]

# todo: 왜 그림이 얼추 나오는 것일까... -_- t+1 일 예측했는데 ...?

# pre.plot(preds_time_minus_1, color="black", is_show=False)
# # pre.plot(preds, color="crimson", is_show=False)
# pre.plot(list(data.data), is_show=True)

pre.plot(pred_finals, color="xkcd:green", is_show=False)
# pre.plot(list(data.data), is_show=True)
pre.plot(list(data.data[next_time:]), color="black", is_show=True)

# --- param (+변화) 확인 --- #
pre.print_param(param)
pre.print_grad(param)






