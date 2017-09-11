import sys
sys.version

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.dates as md
import matplotlib.pyplot as plt
import random
import time
import datetime
import dateutil

import tensorflow as tf
import shutil
import tensorflow.contrib.learn as tflearn
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

np.set_printoptions(threshold=np.nan)


#TF Version
tf.__version__

def get_location_col(p):
    return get_location_col.names[p]

get_location_col.names = {}
get_location_col.names['Timestamp'] = 0

def test_data(series, forecast, num_periods, location=0):
    #print(location)
    # 뒤에서 21번째부터
    test_x_setup = series[-(num_periods + forecast + location):]
    # 20번째까지
    # 20을 num_periods로 변경
    testX = test_x_setup[:num_periods].reshape(-1, num_periods, 1)
    # 뒤에서 20번째부터
    test_y_setup = series[-(num_periods + location):]
    # 20번째까지
    # 20을 num_periods로 변경
    testY = test_y_setup[:num_periods].reshape(-1, num_periods, 1)
    return testX, testY

start = time.time()

option = 'predict'  # start, add, predict
loc_start = 0
loc_end = 100000

#p_hong 의 600 이 압력이 들쭉날쭉하여 트레이닝 하기 힘든 좋은 예

col_name = 'diff_discharge'
cl_name = '2017_06'
cl_al = 'GRU'
num_periods = 100
f_horizon = 1
inputs = 1
hidden = 500
output = 1

learning_rate = 0.001
epochs = 5000
ckpt_name = 'ckpt_' + cl_name + '_' + col_name + '_' + cl_al + '_' + str(num_periods) + '_' + str(hidden) + '_' + str(epochs)
# ckpt_root 에 경로를 집어 넣고 'ckpt_root = ckpt_name'을 주석처리하여 테스트
ckpt_root = ''
ckpt_root = ckpt_name
ckpt_dir = "./"+ckpt_root+"/" + ckpt_name + ".ckpt"

if option == 'start' and not os.path.exists(ckpt_root):
    os.makedirs(ckpt_root)

print('ckpt name', ckpt_name)

#charts = ["p_gwangam","p_oryun_1","p_songpa_1","p_hongtong_1","p_sincheon_1","p_seogang_1","p_lotte_in","p_gimpo","p_yangje","p_gwacheon"]
chart = ["p_songpa_1"]

df = []
df_data_ori = []
df_data_predict = []
df_data_event = []
for i in range(len(chart)):

    print('start')

    col =  chart[i]
    df = pd.read_csv('./ckpt_2017_06_' + col + '_GRU_100_500_5000.csv', dtype=str)
    df.columns = ["Timestamp", col, col + '_pre']

    print('point 1')

    if i == 0:
        df_data_time = df.iloc[:, get_location_col('Timestamp')].values[loc_start: loc_end]

    print('point 2')

    # 실제값
    df_data_tmp = df.iloc[:, 1].values
    df_data_tmp = df_data_tmp.astype(float)
    df_data_ori.append(df_data_tmp[loc_start: loc_end])

    # 예상값
    df_data_tmp = df.iloc[:, 2].values
    df_data_tmp = df_data_tmp.astype(float)
    df_data_predict.append(df_data_tmp[loc_start: loc_end])

    depres_cnt = 0
    df_data_event.append([0] * (loc_end - loc_start))
    for j in range(len(df_data_ori[i])):
        diff = df_data_predict[i][j] - df_data_ori[i][j]
        if diff > 0.20 and df_data_ori[i][j] != 0:
            print( 'sequence:', j, 'x:', df_data_time[j], 'pre:', df_data_predict[i][j], 'ori:', df_data_ori[i][j], 'diff:', diff)
            depres_cnt += 1
            df_data_event[i][j] = df_data_ori[i][j]
        else:
            df_data_event[i][j] = 0

    print('depres_cnt : ', depres_cnt)
    print('point 3')

    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (30, 14)
    plt.title("Forecast vs Actual", fontsize=14)
    plt.ylim([3.0, 6.0])

    dates = [dateutil.parser.parse(s) for s in df_data_time]
    plt.subplots_adjust(bottom=0.2)
    ax = plt.gca()
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)

    print('point 4')
    print(len(dates))
    print(df_data_ori[i].shape)

    plt.plot(dates, pd.Series(np.ravel(df_data_ori[i])), "bo", markersize=3, label="Actual")
    plt.plot(dates, pd.Series(np.ravel(df_data_predict[i])), "r.", markersize=3, label="Forecast")
    plt.plot(dates, pd.Series(np.ravel(df_data_event[i])), "D", markersize=8, label="Event")
    plt.legend(loc="upper left")
    plt.xlabel("Time Periods")
    plt.gcf().autofmt_xdate()
    fig.set_size_inches(34, 6)
    plt.show()

print('ckpt name : ', ckpt_name)
print('datetime : ', datetime.datetime.now())
print('structure : ', 'GRU')
print('time step size : ', num_periods)
print('hidden : ', hidden)
print('epochs : ', epochs)
print('Total time : %f' % (time.time() - start))

