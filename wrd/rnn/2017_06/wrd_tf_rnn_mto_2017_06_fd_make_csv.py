import sys
sys.version

import tensorflow as tf
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.nan)
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
get_location_col.names['dc_1_1'] = 1
get_location_col.names['dc_1_2'] = 2
get_location_col.names['dc_1_3'] = 3
get_location_col.names['dc_1_4'] = 4
get_location_col.names['dc_total_paldang'] = 5
get_location_col.names['dc_36'] = 6
get_location_col.names['dc_gwangam'] = 7
get_location_col.names['dc_hongtong_1'] = 8
get_location_col.names['dc_hongtong_2'] = 9
get_location_col.names['dc_hongtong_3'] = 10
get_location_col.names['dc_seongsan'] = 11
get_location_col.names['dc_gimpo'] = 12
get_location_col.names['dc_bupyeong'] = 13
get_location_col.names['dc_lotte_in'] = 14
get_location_col.names['dc_yangjecheon'] = 15
get_location_col.names['dc_yangje23'] = 16
get_location_col.names['dc_gwacheon'] = 17
get_location_col.names['STL'] = 18

def test_data(series, forecast, num_periods, location=0):
    test_x_setup = series[-(num_periods + forecast + location):]
    testX = test_x_setup[:num_periods].reshape(-1, num_periods, 1)
    test_y_setup = series[-(num_periods + location):]
    testY = test_y_setup[:num_periods].reshape(-1, num_periods, 1)
    return testX, testY

start = time.time()

option = 'predict'  # start, add, predict
location = 600
loc_start = 0
loc_end = 0

col_names = ['dc_total_paldang','dc_seongsan','dc_gimpo','dc_yangjecheon','dc_gwangam']
col_name = 'flow_diff'
cl_name = '2017_06'
structure = 'LSTM'
computational_graph = 'MTO' # Computational Graph
num_periods = 100
f_horizon = 1
inputs = 1
hidden = 500
output = 1

learning_rate = 0.01
epochs = 5000
ckpt_name = 'ckpt_' + structure + '_' + computational_graph + '_' + str(num_periods) + '_' + str(hidden) + '_' + str(epochs) + '_'  + cl_name + '_' + col_name
# ckpt_root 에 경로를 집어 넣고 'ckpt_root = ckpt_name'을 주석처리하여 테스트
ckpt_root = ''
ckpt_root = ckpt_name
ckpt_dir = "./"+ckpt_root+"/" + ckpt_name + ".ckpt"

if option == 'start' and not os.path.exists(ckpt_root):
    os.makedirs(ckpt_root)

print('ckpt name', ckpt_name)

df=pd.read_csv('./2017_06_f_all.csv', dtype=str)
df.columns = ["Timestamp","dc_1_1","dc_1_2","dc_1_3","dc_1_4","dc_total_paldang","dc_36","dc_gwangam","dc_hongtong_1","dc_hongtong_2","dc_hongtong_3","dc_seongsan","dc_gimpo","dc_bupyeong","dc_lotte_in","dc_yangjecheon","dc_yangje23","dc_gwacheon","STL"]

#print(df.values)

df_data_time = df.iloc[:, get_location_col('Timestamp')]
df_data_time_val = df_data_time.values

df_data_ori = []
for i in range(len(col_names)):
    df_data_ori.append(df.iloc[:, get_location_col(col_names[i])])
    for j in range(len(df_data_ori[i])):
        if (df_data_ori[i][j] == 'Bad') or (df_data_ori[i][j] == 'bad') or (df_data_ori[i][j] == '#VALUE!'):
            df_data_ori[i][j] = 0

for i in range(len(df_data_ori)):
    df_data_ori[i] = df_data_ori[i].values.astype(float)

train_data = df_data_ori[0] - (df_data_ori[1] + df_data_ori[2] + df_data_ori[3] + df_data_ori[4])

#plt.ylim(-30000, 20000)
#plt.plot(train_data, "bo", markersize=3, label="Actual")
#plt.show()


TS = np.array(train_data)
print('TS shape:', TS.shape)

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
