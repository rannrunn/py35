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


#TF Version
tf.__version__

def get_location_col(p):
    return get_location_col.names[p]

get_location_col.names = {}
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

#p_hong 의 600 이 압력이 들쭉날쭉하여 트레이닝 하기 힘든 좋은 예

col_name = 'p_songpa_1'
cl_name = '2017_06'
cl_al = 'LSTM'
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

df=pd.read_csv('./2017_06_p_0_10000.csv', dtype=str)
df.columns = ["Timestamp","STL","p_1_3T","p_1_3T_incheon","36_tie","p_36_4","p_gwangam","p_oryun_1","p_oryun_2","p_oryun_2_tie","p_songpa_1","p_songpa_2","p_songpa_tie","p_hongtong_1","p_hongtong_2","p_sincheon_1","p_sincheon_2","p_sincheon_tie","p_seogang_1","p_seogang_2","p_seogang_tie","p_myeongsudae","p_lotte_in","p_gimpo","p_bupyeong","p_yangje","p_gwacheon"]

#print(df.values)
print(get_location_col(col_name))
# 2, 3-4-5-7, 6
df_data = df.iloc[:, get_location_col(col_name)]
print(df_data)
df_data = df_data.astype(float)
df_data_val = df_data.values
print('array')
print(df_data_val.shape)

TS = np.array(df_data)

x_data = TS[:len(TS)-len(TS) % num_periods]
x_batches = x_data.reshape(-1, num_periods, 1)

y_data = TS[1:(len(TS)-(len(TS) % num_periods))+f_horizon]
y_batches = y_data.reshape(-1, num_periods, 1)

X_test, Y_test = test_data(TS, f_horizon, num_periods, location)

print(x_batches.shape)
print(y_batches.shape)

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, num_periods, inputs])
y = tf.placeholder(tf.float32, [None, num_periods, output])

basic_cell = tf.contrib.rnn.LSTMCell(num_units=hidden)
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
stacked_outputs = tf.layers.dense(rnn_output, output)
loss = tf.reduce_sum(tf.square(stacked_outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)

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

    writer = tf.summary.FileWriter('c:/rnn', sess.graph)

    try:
        if (option != 'predict'):
            for ep in range(epochs):
                sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
                if ep % 100 == 0:
                    mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
                    print(ep, "\tMSE:", mse)

                if ep != 0 and ep % 5000 == 0:
                    # Save the variables to disk.
                    save_path = saver.save(sess, ckpt_dir)
                    print("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            save_path = saver.save(sess, ckpt_dir)
            print("Model saved in file: %s" % save_path)

        # outputs를 삭제하고 stacked_outputs으로 변경
        y_pred = sess.run(stacked_outputs, feed_dict={X: X_test})
        pred_mse = loss.eval(feed_dict={X: X_test, y: Y_test})
        print('pred_mse : ', pred_mse)
        #print(y_pred)
        sess.close()
    except Exception as ex:
        print('Exception sess run : ', ex)
        pass
    finally:
        sess.close()

print('ckpt name : ', ckpt_name)
print('datetime : ', datetime.datetime.now())
print('structure : ', 'LSTM')
print('time step size : ', num_periods)
print('hidden : ', hidden)
print('epochs : ', epochs)
print('Total time : %f' % (time.time() - start))

plt.title("Forecast vs Actual", fontsize=14)
plt.plot(pd.Series(np.ravel(Y_test)), "bo", markersize=10, label="Actual")
plt.plot(pd.Series(np.ravel(y_pred)), "r.", markersize=10, label="Forecast")
plt.xlabel("Time Periods")

plt.show()

