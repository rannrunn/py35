import sys
sys.version

import tensorflow as tf
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.nan)
import random
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

import csv

#TF Version
tf.__version__

def get_location_col(p):
    return get_location_col.names[p]

get_location_col.names = {}
get_location_col.names['Timestamp'] = 0
get_location_col.names['STL'] = 1
get_location_col.names['p_1_3T'] = 2
get_location_col.names['p_1_3T_incheon'] = 3
get_location_col.names['36_tie'] = 4
get_location_col.names['p_36_4'] = 5
get_location_col.names['p_gwangam'] = 6
get_location_col.names['p_oryun_1'] = 7
get_location_col.names['p_oryun_2'] = 8
get_location_col.names['p_oryun_2_tie'] = 9
get_location_col.names['p_songpa_1'] = 10
get_location_col.names['p_songpa_2'] = 11
get_location_col.names['p_songpa_tie'] = 12
get_location_col.names['p_hongtong_1'] = 13
get_location_col.names['p_hongtong_2'] = 14
get_location_col.names['p_sincheon_1'] = 15
get_location_col.names['p_sincheon_2'] = 16
get_location_col.names['p_sincheon_tie'] = 17
get_location_col.names['p_seogang_1'] = 18
get_location_col.names['p_seogang_2'] = 19
get_location_col.names['p_seogang_tie'] = 20
get_location_col.names['p_myeongsudae'] = 21
get_location_col.names['p_lotte_in'] = 22
get_location_col.names['p_gimpo'] = 23
get_location_col.names['p_bupyeong'] = 24
get_location_col.names['p_yangje'] = 25
get_location_col.names['p_gwacheon'] = 26

def test_data(series, forecast, num_periods, location=0):
    test_x_setup = series[-(num_periods + forecast + location):]
    testX = test_x_setup[:num_periods].reshape(-1, num_periods, 1)
    test_y_setup = series[-(num_periods + location):]
    testY = test_y_setup[:num_periods].reshape(-1, num_periods, 1)
    return testX, testY

start = time.time()

option = 'predict'  # start, add, predict
location = 3000

#p_hong 의 600 이 압력이 들쭉날쭉하여 트레이닝 하기 힘든 좋은 예

col_name = 'p_songpa_1'
cl_name = '2017_06'
structure = 'LSTM'
computational_graph = 'MTO' # Computational Graph
num_periods = 100
f_horizon = 1
inputs = 1
hidden = 500
output = 1

learning_rate = 0.001
epochs = 25000
ckpt_name = 'ckpt_' + structure + '_' + computational_graph + '_' + str(num_periods) + '_' + str(hidden) + '_' + str(epochs) + '_'  + cl_name + '_' + col_name
# ckpt_root 에 경로를 집어 넣고 'ckpt_root = ckpt_name'을 주석처리하여 테스트
ckpt_root = ''
ckpt_root = ckpt_name
ckpt_dir = "./"+ckpt_root+"/" + ckpt_name + ".ckpt"

if option == 'start' and not os.path.exists(ckpt_root):
    os.makedirs(ckpt_root)

print('ckpt name', ckpt_name)

df=pd.read_csv('./2017_06_p_all.csv', dtype=str)
df.columns = ["Timestamp","STL","p_1_3T","p_1_3T_incheon","36_tie","p_36_4","p_gwangam","p_oryun_1","p_oryun_2","p_oryun_2_tie","p_songpa_1","p_songpa_2","p_songpa_tie","p_hongtong_1","p_hongtong_2","p_sincheon_1","p_sincheon_2","p_sincheon_tie","p_seogang_1","p_seogang_2","p_seogang_tie","p_myeongsudae","p_lotte_in","p_gimpo","p_bupyeong","p_yangje","p_gwacheon"]

#print(df.values)

df_data_time = df.iloc[:, get_location_col('Timestamp')]
df_data_time_val = df_data_time.values

df_data_ori = df.iloc[:, get_location_col(col_name)]
for i in range(len(df_data_ori)):
    if(df_data_ori[i] == 'Bad') or (df_data_ori[i] == 'bad'):
        df_data_ori[i] = 0

df_data_ori = df_data_ori.astype(float)
df_data_ori_val = df_data_ori.values

print('array')
print(df_data_time_val.shape)

TS = np.array(df_data_ori)

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, num_periods, inputs])
y = tf.placeholder(tf.float32, [None, 1])

cell = tf.contrib.rnn.LSTMCell(num_units=hidden, activation=tf.tanh)
rnn_output, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(rnn_output[:, -1], output, activation_fn=None)
loss = tf.reduce_sum(tf.square(Y_pred - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

print('before sess')
with tf.Session() as sess:
    print('session run')

    saver = tf.train.Saver()

    saver.restore(sess, ckpt_dir)
    print("Model restored.")

    try:

        csvfile = open(ckpt_name + '.csv', 'w', newline='')
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['Timestamp', col_name, col_name + '_pre'])


        plt_x_input = []
        plt_y_input = []
        plt_y_pred = []
        for i in range(len(TS) - num_periods, 0, -1):
            location = i
            X_test, Y_test = test_data(TS, f_horizon, num_periods, location)
            test_predict = sess.run(Y_pred, feed_dict={X: X_test})

            y_input = np.ravel(Y_test)[-1:]

            spamwriter.writerow([str(df_data_time_val[len(TS) - i - f_horizon]), y_input[0], test_predict[0][0]])

            if i % 100 == 0:
                print('index : ', i)

        print('ckpt name : ', ckpt_name)
        print('datetime : ', datetime.datetime.now())
        print('structure : ', structure)
        print('computational graph : ', computational_graph)
        print('time step size : ', num_periods)
        print('hidden : ', hidden)
        print('epochs : ', epochs)
        print('Total time : %f' % (time.time() - start))

        sess.close()
    except Exception as ex:
        print('Exception sess run : ', ex)
        pass
    finally:
        sess.close()



