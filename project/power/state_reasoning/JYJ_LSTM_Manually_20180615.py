import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import random
import math
import os
import matplotlib.pyplot as plt
import copy
import JYJ_preprocessing_LSTM as pre     # todo: 여기 밑줄 왜 뜨지...?
from tqdm import tqdm

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

    # data_x = pd.read_csv("D:\\buksiwha_pqms_phase_3.csv")
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
    #
    # pre.plot(data_x)
    # pre.plot(data_x_denormalized)
    # pre.plot(data_x_normalized)
    # pre.head(data_x, 10)

    # 3) sine data

    # Sine 데이터 생성
    data_x = np.sin(np.arange(0, 30, 0.1)).tolist()
    # 데이터 노멀라이즈
    data_x, max_data, min_data = pre.normalize(data_x, -0.5, 0.5)
    # 생성한 Sine 데이터 출력
    pre.plot('1', data_x)

    # --- preprocess data --- #
    # ?
    n_col = 1
    # Time Step Size
    n_row_input = 70
    # 출력 사이즈
    n_row_output = 1
    # ?
    n_row = pre.get_n_row(data_x)

    # make None value on 'data_x'
    # 생성한 데이터 중 10% 데이터를 np.nan으로 대체
    rand_idx = np.random.choice(np.arange(1, n_row, 1), size=n_row//10, replace=False)
    for idx in rand_idx:
        data_x[idx] = np.nan

    # RNN의 입력에 사용할 데이터와 출력에 사용할 데이터 생성
    # data_for_rnn_x : 길이 230, 개별 인자별 길이 70
    # data_for_rnn_y : 길이 230, 개별 인자별 길이 1
    data_for_rnn_x, data_for_rnn_y = pre.get_list_type_data(data_x, n_row_input, n_row_output)

    print('')
    print('data_for_rnn_x:', len(data_for_rnn_x[0]))
    print('data_for_rnn_y', len(data_for_rnn_y[0]))
    print('')

    # --- train setting --- #
    # todo: pre 로 받아온 함수들 정리 ㅡ preprocessing 과 lstm 관련쪽 나눔
    param = pre.Param()
    print('param:', param)
    # optimizer = torch.optim.SGD(pre.parameters(param), lr=0.01)

    learning_rate = 0.01
    optimizer = torch.optim.Adam(pre.parameters(param), lr=learning_rate)
    # epoch 사이즈
    num_iter = 15
    n = len(data_for_rnn_x)

    # --- train --- #
    # todo: 학습 잘 안됨...! 왜 그럴까...?

    # todo: loss surface 확인
    nan_idxs = []
    loss_arr = []

    for iter_ in tqdm(range(num_iter)):

        # Learning rate decay
        # 매 5회 iteration 마다 learning rate 를 0.01 씩 증가
        if (iter_+1) % 5 == 0:
            learning_rate *= 0.01
            optimizer = torch.optim.Adam(pre.parameters(param), lr=learning_rate)
            print("Learning Rate:", learning_rate)

        cnt = 0

        for idx in range(n):
            cnt += 1
            data = data_for_rnn_x[idx]
            # y = data_for_rnn_y[idx]

            if idx == 0:
                outputs = data

            output, outputs = pre.lstm_forward_propagation_with_missing_data(outputs, data, param, verbose=False)
            # todo: 데이터 몇개 비우고 이걸로 해봄 !!!!!!!!!!!!!!!!!!!!!!!!!!!
            # todo: 여기에 D 도 추가 !!!!!!!!!!!!!!!!!!!
            # todo: 이상한 논리 제거 !!!!!!!!!!!!!!!!!!!!

            # output, outputs = pre.lstm_forward_propagation(data, param, verbose=False)
            # todo: 학습이 잘 안되는 이유는.. t 시점의 값을 예측하는 것과, t+1 시점을 예측하는 것의 차이 아닐까 ...?
            # todo: 이 부분을 고려해서 ㅡ 구조를 변경해야 할 듯
            # todo: 이게 진짜 문제인지 ㅡ 맞는지 틀린지 정확히 모르겠음 ㅡ 생각 필요

            # loss = pre.loss_fn(y=y, pred=output)

            # nan 데이터가 아닐 경우 losses 리스트에 MSE 값을 append
            # data의 길이는 70
            losses = []
            for idx in range(len(data)):
                if not pre.isnan(data[idx]):
                    losses.append(pre.loss_fn(data[idx], outputs[idx]))

            # loss 를 모두 더함
            loss = sum(losses)

            # 기울기 초기화
            param = pre.zero_grad(param)  # set gradients as 0
            # back propagation 진행
            loss.backward(retain_graph=True)  # compute gradient of the loss
            # 옵티마이저를 이용해 가중치 업데이트를 진행
            optimizer.step()    # apply back propagation

            # 더한 손실값을 소수점 셋째자리에서 반올림
            loss_val = round(loss.data[0], 3)


            if cnt % 100 == 0:
                print(loss_val)   # todo: 학습 잘 되다가 왜 nan 이 뜰까 ...? SGD 로 학습할 경우
            loss_arr.append(loss_val)

            if np.isnan(loss.data[0]):
                nan_idxs.append([iter_, idx])
                break

pre.plot('2', loss_arr)

pre.plot('3', data_x)

for idx in range(n):
    data = data_for_rnn_x[idx]
    y = data_for_rnn_y[idx]
    if idx == 0:
        preds = data
    y_pred, preds = pre.lstm_forward_propagation_with_missing_data(preds, data, param, verbose=False)

    if idx % 10 == 0:
        break

preds = [x.data[0] for x in preds]
print('y_pred', len(y_pred))
print('preds', len(preds))
print('y_pred.data', y_pred.data)

pre.plot('4', preds, is_show=True)
pre.plot('5', list(data.data), is_show=True)


pre.plot('6', list(data.data) + list(y.data), is_show=True)
pre.plot('7', list(data.data) + list(y_pred.data), is_show=True)


# --- param 변화 확인 --- #
print('파람 변화 확인')
pre.print_param(param)
print('기울기 변화 확인')
pre.print_grad(param)







