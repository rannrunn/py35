# coding: utf-8
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
get_location_col.names['f_paldang'] = 1
get_location_col.names['f_36t'] = 2
get_location_col.names['f_total'] = 3
get_location_col.names['f_gwangam'] = 4
get_location_col.names['f_hongtong'] = 5
get_location_col.names['f_seongsan_gimpo'] = 6
get_location_col.names['f_yangjecheon'] = 7
get_location_col.names['f_yangje_23t'] = 8
get_location_col.names['f_gwacheon_ga'] = 9
get_location_col.names['f_lotte_in'] = 10
get_location_col.names['STL'] = 11
get_location_col.names['p_3t_1dan'] = 12
get_location_col.names['p_3t_incheon_1dan'] = 13
get_location_col.names['p_gwangam'] = 14
get_location_col.names['g_36t'] = 15
get_location_col.names['p_36t_4dan'] = 16
get_location_col.names['p_oryun_1'] = 17
get_location_col.names['p_oryun_2'] = 18
get_location_col.names['p_oryun_t'] = 19
get_location_col.names['p_songpa_1'] = 20
get_location_col.names['p_songpa_2'] = 21
get_location_col.names['p_songpa_t'] = 22
get_location_col.names['p_lotte_in'] = 23
get_location_col.names['p_sincheon_1'] = 24
get_location_col.names['p_sincheon_2'] = 25
get_location_col.names['p_sincheon_t'] = 26
get_location_col.names['p_hongtong_1'] = 27
get_location_col.names['p_hongtong_2'] = 28
get_location_col.names['p_seogang_1'] = 29
get_location_col.names['p_seogang_2'] = 30
get_location_col.names['p_seogang_t'] = 31
get_location_col.names['p_yangjecheon'] = 32
get_location_col.names['p_gwacheon_ga'] = 33
get_location_col.names['p_3t_2dan'] = 34
get_location_col.names['p_yangje_23t'] = 35

start = time.time()

option = 'start'  # start, add, predict

# p_gwangam, p_oryun_1, p_songpa_1, p_sincheon_1, p_hongtong_1, p_seogang_1
col_names = ["f_hongtong","p_lotte_in","p_oryun_1","p_songpa_1","p_oryun_2","p_songpa_2"]
col_name = ""
for i in range(len(col_names)):
    col_name = ""
cl_name = '170821_bunseok'

learning_rate = 0.01
epochs = 50000
ckpt_name = 'ckpt_' + str(epochs) + '_'  + cl_name + col_name
# ckpt_root 에 경로를 집어 넣고 'ckpt_root = ckpt_name'을 주석처리하여 테스트
ckpt_root = ''
ckpt_root = ckpt_name
ckpt_dir = "./"+ckpt_root+"/" + ckpt_name + ".ckpt"

if option == 'start' and not os.path.exists(ckpt_root):
    os.makedirs(ckpt_root)

print('ckpt name', ckpt_name)

df=pd.read_csv('./170821_bunseok.csv', dtype=str)
df.columns = ["Timestamp","f_paldang","f_36t","f_total","f_gwangam","f_hongtong","f_seongsan_gimpo","f_yangjecheon","f_yangje_23t","f_gwacheon_ga","f_lotte_in","STL","p_3t_1dan","p_3t_incheon_1dan","p_gwangam","g_36t","p_36t_4dan","p_oryun_1","p_oryun_2","p_oryun_t","p_songpa_1","p_songpa_2","p_songpa_t","p_lotte_in","p_sincheon_1","p_sincheon_2","p_sincheon_t","p_hongtong_1","p_hongtong_2","p_seogang_1","p_seogang_2","p_seogang_t","p_yangjecheon","p_gwacheon_ga","p_3t_2dan","p_yangje_23t"]

#print(df.values)

df_data_time = df.iloc[:, get_location_col('Timestamp')]
df_data_time_val = df_data_time.values

df_data_ori = []
for i in range(len(col_names)):
    df_data_ori.append(df.iloc[:1000, get_location_col(col_names[i])])
    for j in range(len(df_data_ori[i])):
        if (df_data_ori[i][j] == 'Bad') or (df_data_ori[i][j] == 'bad') or (df_data_ori[i][j] == '#VALUE!'):
            df_data_ori[i][j] = 0

for i in range(len(df_data_ori)):
    df_data_ori[i] = df_data_ori[i].values.astype(float)

train_data_1 = df_data_ori[0] + df_data_ori[1]
train_data_2 = df_data_ori[2] - df_data_ori[3]
train_data_3 = df_data_ori[4] - df_data_ori[5]

print('111111')
print(train_data_1)
print('222222')
print(train_data_2)
print('333333')
print(train_data_3)

x_data_1 = train_data_2
x_data_2 = train_data_3
y_data = train_data_1
# W는 설명변수, b는 보정값
W1 = tf.Variable(tf.random_uniform([1], -0.1, 1.0))
W2 = tf.Variable(tf.random_uniform([1], -0.1, 1.0))
b = tf.Variable(tf.random_uniform([1], -0.1, 1.0))

X1 = tf.placeholder(tf.float32)
X2 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
# 방정식 모델입니다.
hypothesis = W1 * X1 + b
# cost 함수입니다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Gradient Descent
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)


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

    print(sess.run(W1))
    print(sess.run(W2))
    print(sess.run(b))

    plt.title("Forecast vs Actual", fontsize=14)
    plt.plot(pd.Series(np.ravel(x_data_1)), "bo", markersize=10, label="Actual")
    plt.plot(pd.Series(np.ravel(y_data)), "r.", markersize=10, label="Forecast")
    plt.xlabel("Time Periods")

    try:
        if (option != 'predict'):
            for step in range(epochs):
                sess.run(train, feed_dict={X1: x_data_1, Y: y_data})
                if step % 200 == 0:
                    step_loss = sess.run(cost, feed_dict={X1: x_data_1, Y: y_data})
                    print(step, step_loss, sess.run(W1), sess.run(W2), sess.run(b))
                if step != 0 and step % 5000 == 0:
                    # Save the variables to disk.
                    save_path = saver.save(sess, ckpt_dir)
                    print("Model saved in file: %s" % save_path)
            # Save the variables to disk.
            save_path = saver.save(sess, ckpt_dir)
            print("Model saved in file: %s" % save_path)


            print(sess.run(hypothesis, feed_dict={X1: 5.0, X2: 2.5}))


        print('ckpt name : ', ckpt_name)
        print('datetime : ', datetime.datetime.now())
        print('epochs : ', epochs)
        print('Total time : %f' % (time.time() - start))



        plt.show()

        sess.close()
    except Exception as ex:
        print('Exception sess run : ', ex)
        pass
    finally:
        sess.close()


