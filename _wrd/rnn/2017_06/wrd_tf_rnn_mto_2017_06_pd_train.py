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

start = time.time()

option = 'start'  # start, add, predict
location = 600
loc_start = 0
loc_end = 0

# p_gwangam, p_oryun_1, p_songpa_1, p_sincheon_1, p_hongtong_1, p_seogang_1
col_names = ["p_oryun_1","p_songpa_1"]
col_name = 'diff_' + col_names[0] + '_' + col_names[1]
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

df=pd.read_csv('./2017_06_p_0_10000.csv', dtype=str)
df.columns = ["Timestamp","STL","p_1_3T","p_1_3T_incheon","36_tie","p_36_4","p_gwangam","p_oryun_1","p_oryun_2","p_oryun_2_tie","p_songpa_1","p_songpa_2","p_songpa_tie","p_hongtong_1","p_hongtong_2","p_sincheon_1","p_sincheon_2","p_sincheon_tie","p_seogang_1","p_seogang_2","p_seogang_tie","p_myeongsudae","p_lotte_in","p_gimpo","p_bupyeong","p_yangje","p_gwacheon"]

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

train_data = df_data_ori[0] - df_data_ori[1]

#plt.ylim(-30000, 20000)
#plt.plot(train_data, "bo", markersize=3, label="Actual")
#plt.show()


TS = np.array(train_data)
print('TS shape:', TS.shape)

x_data = []
y_data = []
for i in range(0 + int(random.random() * 10), len(TS) - num_periods, num_periods):
    _x = TS[i:i + num_periods]
    _y = TS[i + num_periods]
    x_data.append(_x)
    y_data.append(_y)
    pass

x_batches = np.array(x_data).reshape(-1, num_periods, 1)
y_batches = np.array(y_data).reshape(-1, 1)

print('x_batches shape:',x_batches.shape)
print('y_batches shape:',y_batches.shape)

XX_test = []
YY_test = []
for i in range(location, location + num_periods + num_periods + num_periods):
    _x = TS[i:i + num_periods]
    _y = TS[i + num_periods]
    XX_test.append(_x)
    YY_test.append(_y)
    pass

X_test = np.array(XX_test).reshape(-1, num_periods, 1)
Y_test = np.array(YY_test).reshape(-1, 1)

print('X_test shape:',X_test.shape)
print('Y_test shape:',Y_test.shape)

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
    if (option == 'start'):
        init = tf.global_variables_initializer()
        init.run()
    elif (option == 'add' or option == 'predict'):
        # Restore variables from disk.
        saver.restore(sess, ckpt_dir)
        print("Model restored.")
    try:
        if (option != 'predict'):
            for ep in range(epochs):
                _, step_loss = sess.run([training_op, loss], feed_dict={X: x_batches, y: y_batches})
                if ep % 100 == 0:
                    print("[step: {}] loss: {}".format(ep, step_loss))
                if ep != 0 and ep % 5000 == 0:
                    # Save the variables to disk.
                    save_path = saver.save(sess, ckpt_dir)
                    print("Model saved in file: %s" % save_path)
            # Save the variables to disk.
            save_path = saver.save(sess, ckpt_dir)
            print("Model saved in file: %s" % save_path)

        # outputs를 삭제하고 stacked_outputs으로 변경
        test_predict = sess.run(Y_pred, feed_dict={X: X_test})
        rmse_val = sess.run(rmse, feed_dict={targets: Y_test, predictions: test_predict})
        print('pred_mse : ', rmse_val)
        #print(y_pred)

        print('ckpt name : ', ckpt_name)
        print('datetime : ', datetime.datetime.now())
        print('structure : ', structure)
        print('computational graph : ', computational_graph)
        print('time step size : ', num_periods)
        print('hidden : ', hidden)
        print('epochs : ', epochs)
        print('Total time : %f' % (time.time() - start))

        plt.title("Forecast vs Actual", fontsize=14)
        plt.plot(pd.Series(np.ravel(Y_test)), "bo", markersize=10, label="Actual")
        plt.plot(pd.Series(np.ravel(test_predict)), "r.", markersize=10, label="Forecast")
        plt.xlabel("Time Periods")

        plt.show()

        sess.close()
    except Exception as ex:
        print('Exception sess run : ', ex)
        pass
    finally:
        sess.close()


