import sys
sys.version

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import random
import time
import datetime

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
get_location_col.names['p_gwangam'] = 2
get_location_col.names['p_oryun'] = 3
get_location_col.names['p_songpa'] = 4
get_location_col.names['p_sincheon'] = 5
get_location_col.names['p_hong'] = 6
get_location_col.names['p_seongsan'] = 7

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
location = 600
print_location = False

#p_hong 의 600 이 압력이 들쭉날쭉하여 트레이닝 하기 힘든 좋은 예

col_name = 'p_songpa'
cl_name = 'lotte'
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

df=pd.read_csv('./wrd_clock_pres_chart.csv', dtype=str)
df.columns = ["state", "datetime", "p_gwangam", "p_oryun", "p_songpa", "p_sincheon", "p_hong", "p_seongsan"]

print(df.values)

# 2, 3-4-5-7, 6
df_data = df.iloc[:, get_location_col(col_name)]
df_data = df_data.astype(float)
df_data_val = df_data.values
print('array')
print(df_data_val.shape)



# random : seed 값을 주면 일정한 랜덤값이 나온다. 필요없어서 주석 처리합니다.
# random.seed(111)
# rng = pd.date_range(start='2000', periods=1441, freq='M')
# pd.Series  에서 rng는 index 값으로 들어간 것
# cumsum 은 요청된 축에 대한 누적합계를 반환합니다.
# ts  = pd.Series(df_data_val, rng)
# ts.plot(c='r', title='Example Time Series')
# plt.show()
# print(ts.head(10))


# 테스트를 위한 코드
# rng = pd.date_range(start='2000', periods=1441, freq='M')
# ts  = pd.Series(np.random.uniform(-0.010, 0.010, size=len(rng)), rng).cumsum()


TS = np.array(df_data)
# print(TS.shape) # (209,)


x_data = TS[:len(TS)-len(TS) % num_periods]
# 20을 num_periods로 변경
x_batches = x_data.reshape(-1, num_periods, 1)

y_data = TS[1:(len(TS)-(len(TS) % num_periods))+f_horizon]
# 20을 num_periods로 변경
y_batches = y_data.reshape(-1, num_periods, 1)

# print(len(x_batches))
# print(x_batches.shape)
# print(x_batches[0:2])

# print(y_batches[0:1])
# print(y_batches.shape)

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, num_periods, inputs])
y = tf.placeholder(tf.float32, [None, num_periods, output])

#BasicRNNCell -> GRUCell 로 바꾸고 activation=tf.nn.relu 값은 삭제
basic_cell = tf.contrib.rnn.GRUCell(num_units=hidden)
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

# stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
# stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
# 내 생각에는 하나로 줄일 수 있을 것 같아 줄인다.
stacked_outputs = tf.layers.dense(rnn_output, output)

# 필요 없을 것 같아 삭제
# outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])
# reduce_sum 총합
# outputs을 삭제하고 stacked_outputs으로 변경
loss = tf.reduce_sum(tf.square(stacked_outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)

with tf.Session() as sess:
    print('session run')

    saver = tf.train.Saver()

    saver.restore(sess, ckpt_dir)
    print("Model restored.")
    try:
        print('TS length : ', len(TS))
        plt_y_input = []
        plt_y_pred = []
        plt_y_event = []

        for i in range(len(TS) - num_periods, 0, -1):
            #print('index : ', i)

            if print_location == True:
                print('print_location == True')
            else:
                location = i

            X_test, Y_test = test_data(TS, f_horizon, num_periods, location)
            y_pred = sess.run(stacked_outputs, feed_dict={X: X_test})

            #print(np.ravel(Y_test))
            #print(np.ravel(Y_test)[-1:])
            #print(np.ravel(y_pred))
            #print(np.ravel(y_pred)[-1:])

            y_input = np.ravel(Y_test)[-1:]
            y_pred = np.ravel(y_pred)[-1:]

            plt_y_input = np.append(plt_y_input, y_input)
            plt_y_pred = np.append(plt_y_pred, y_pred)
            plt_y_event = np.append(plt_y_event, [0])

            pred_mse = loss.eval(feed_dict={X: X_test, y: Y_test})
            #print('pred_mse : ', pred_mse)


            abs_y = abs(y_pred - y_input)
            #print(abs_y)

            if i % 100 == 0:
                print('index : ', i)

            if y_pred - y_input > 0.16:
                print('event index : ', i)
                print(abs_y)
                plt_y_event[-1:] = y_input

        #print(plt_y_input.shape)
        #print(np.ravel(plt_y_input))
        #print(plt_y_pred.shape)
        #print(np.ravel(plt_y_pred))

        #print(plt_y_event)

        plt.title("Forecast vs Actual", fontsize=14)
        plt.ylim([4.5, 6.0])
        # np.ravel => 1차 행렬로 변환해 준다. order를 옵션으로 줄 수 있다.
        plt.plot(pd.Series(np.ravel(plt_y_event)), "D", markersize=8, label="Event")
        plt.plot(pd.Series(np.ravel(plt_y_input)), "bo", markersize=3, label="Actual")
        # plt.plot(pd.Series(np.ravel(Y_test)), "w*", markersize=10)
        plt.plot(pd.Series(np.ravel(plt_y_pred)), "r.", markersize=3, label="Forecast")
        plt.legend(loc="upper left")
        plt.xlabel("Time Periods")


        plt.show()

        #print(y_pred)
        sess.close()
    except Exception as ex:
        print('Exception sess run : ', ex)
        pass
    finally:
        sess.close()

print('ckpt name : ', ckpt_name)
print('datetime : ', datetime.datetime.now())
print('structure : ', 'GRU')
print('time step size : ', num_periods)
print('hidden : ', hidden)
print('epochs : ', epochs)
print('Total time : %f' % (time.time() - start))

