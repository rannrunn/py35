# coding: utf-8
import sys
sys.version
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.nan)
import os
import time
import datetime
from non_linear_regression.function import correlation
import tensorflow as tf
import traceback


def get_location_col(p):
    return get_location_col.names[p]

def main(conn):

    try:

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
        get_location_col.names['p_36t'] = 15
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


        col_names_hongtong_lotte_1 = ["Timestamp","f_hongtong","f_lotte_in","p_3t_1dan","p_3t_incheon_1dan","p_3t_1dan","p_36t","p_3t_1dan","p_oryun_1"]
        # 0:Timestamp Indes, 1:Flow Start Index, 2:Flow End Index, 3:Pressure Strat Index, 4:Pressure End Index
        col_names_hongtong_lotte_1_idx = [0,1,2,3,8]
        col_names_hongtong_lotte_1_x_data_cnt = 3
        col_names_hongtong_lotte_2 = ["Timestamp","f_hongtong","f_lotte_in","p_oryun_1","p_songpa_1","p_oryun_2","p_songpa_2"]
        col_names_hongtong_lotte_2_idx = [0,1,2,3,6]
        col_names_hongtong_lotte_2_x_data_cnt = 2
        col_names_hongtong_1 = ["Timestamp","f_hongtong","p_songpa_1","p_sincheon_1","p_songpa_1","p_sincheon_2","p_songpa_1","p_lotte_in"]
        col_names_hongtong_1_idx = [0,1,1,2,7]
        col_names_hongtong_1_x_data_cnt = 3
        col_names_hongtong_2 = ["Timestamp","f_hongtong","p_sincheon_1","p_hongtong_1","p_sincheon_2","p_hongtong_2"]
        col_names_hongtong_2_idx = [0,1,1,2,5]
        col_names_hongtong_2_x_data_cnt = 2

        idx_col = col_names_hongtong_lotte_2_idx
        col_names = col_names_hongtong_lotte_2
        cl_name = '170821_analysis_hongtong_lotte_2'

        idx_time = idx_col[0]
        idx_f_start = idx_col[1]
        idx_f_end = idx_col[2]
        idx_p_start = idx_col[3]
        idx_p_end = idx_col[4]

        col_name = ""
        for i in range(len(col_names)):
            if i == 0:
                col_name = col_names[i]
            else:
                col_name += '_' + col_names[i]

        learning_rate_weight = 10000
        learning_rate_bias = 0.00001
        start = 45000
        end = 100000
        epochs = 1000000
        difference = 4000
        ckpt_name = 'ckpt_'  + cl_name
        # ckpt_root 에 경로를 집어 넣고 'ckpt_root = ckpt_name'을 주석처리하여 테스트
        ckpt_root = ''
        ckpt_root = ckpt_name
        ckpt_dir = "./"+ckpt_root+"/" + ckpt_name + ".ckpt"

        if option == 'start' and not os.path.exists(ckpt_root):
            os.makedirs(ckpt_root)

        print('ckpt name', ckpt_name)

        df=pd.read_csv('./170821_analysis.csv', dtype=str)
        df.columns = ["Timestamp","f_paldang","f_36t","f_total","f_gwangam","f_hongtong","f_seongsan_gimpo","f_yangjecheon","f_yangje_23t","f_gwacheon_ga","f_lotte_in","STL","p_3t_1dan","p_3t_incheon_1dan","p_gwangam","p_36t","p_36t_4dan","p_oryun_1","p_oryun_2","p_oryun_t","p_songpa_1","p_songpa_2","p_songpa_t","p_lotte_in","p_sincheon_1","p_sincheon_2","p_sincheon_t","p_hongtong_1","p_hongtong_2","p_seogang_1","p_seogang_2","p_seogang_t","p_yangjecheon","p_gwacheon_ga","p_3t_2dan","p_yangje_23t"]

        df_data_time = df.iloc[:, get_location_col('Timestamp')]
        df_data_time_val = df_data_time.values[start:end]

        df_ext = df[col_names]
        df_ext = df_ext[start:end]
        for i in range(1, len(col_names), 1):
            print('xxxx:',i)
            #print((df_ext[col_names[i]] != 'Bad'))
            df_ext = df_ext[(df_ext[col_names[i]] != 'Bad')]
            df_ext = df_ext[(df_ext[col_names[i]] != 'bad')]
            df_ext = df_ext[(df_ext[col_names[i]] != '#VALUE!')]

        for i in range(1, len(col_names), 1):
            df_ext[col_names[i]] = df_ext[col_names[i]].astype(float)

        df_ext["x_data_1"] = df_ext[col_names[idx_col[3]]] - df_ext[col_names[idx_col[3] + 1]]
        df_ext["x_data_2"] = df_ext[col_names[idx_col[3] + 2]] - df_ext[col_names[idx_col[3] + 3]]

        for i in range(idx_col[1], idx_col[2] + 1, 1):
            if i == idx_col[1]:
                df_ext["y_data"] = df_ext[col_names[i]]
            else:
                df_ext["y_data"] = df_ext["y_data"] + df_ext[col_names[i]]

        df_ext = df_ext[(df_ext['x_data_1'] > 0)]
        df_ext = df_ext[(df_ext['x_data_2'] > 0)]
        df_ext = df_ext[(df_ext['y_data'] > 0)]

        data_time = df_ext["Timestamp"].values
        x_data_1 = df_ext["x_data_1"].values
        x_data_2 = df_ext["x_data_2"].values
        y_data = df_ext["y_data"].values


        W1 = tf.Variable(tf.random_uniform([1], 90000000, 100000000), name='weight_01', dtype=tf.float32)
        W2 = tf.Variable(tf.random_uniform([1], 90000000, 100000000), name='weight_02', dtype=tf.float32)
        b = tf.Variable(tf.random_uniform([1], 0, 0), name='bias', dtype=tf.float32)

        X1 = tf.placeholder(tf.float32)
        X2 = tf.placeholder(tf.float32)
        Y = tf.placeholder(tf.float32)

        list_weight = []
        list_weight.append(W1)
        list_weight.append(W2)

        list_bias = []
        list_bias.append(b)

        hypothesis = tf.sqrt(X1 * W1) + tf.sqrt(X2 * W2) + b
        cost_mse = tf.reduce_mean(tf.square(hypothesis - y_data))
        cost_abs = tf.reduce_mean(tf.abs(hypothesis - y_data))
        train_op1 = tf.train.GradientDescentOptimizer(learning_rate_weight).minimize(cost_mse, var_list=list_weight)
        train_op2 = tf.train.GradientDescentOptimizer(learning_rate_bias).minimize(cost_mse, var_list=list_bias)
        train_op = tf.group(train_op1, train_op2)


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

            try:
                if (option != 'predict'):
                    for step in range(epochs):
                        sess.run(train_op, feed_dict={X1: x_data_1, X2: x_data_2, Y: y_data})
                        if step % 200 == 0:
                            step_loss = sess.run(cost_mse, feed_dict={X1: x_data_1, X2: x_data_2, Y: y_data})
                            conn.send("xxxx1".encode())
                            print(step, step_loss, sess.run(W1), sess.run(W2), sess.run(b))
                        if step != 0 and step % 5000 == 0:
                            # Save the variables to disk.
                            save_path = saver.save(sess, ckpt_dir)
                            print("Model saved in file: %s" % save_path)
                    # Save the variables to disk.
                    save_path = saver.save(sess, ckpt_dir)
                    print("Model saved in file: %s" % save_path)


                predict_data, predict_cost_mse, predict_cost_abs = sess.run([hypothesis, cost_mse, cost_abs], feed_dict={X1: x_data_1, X2: x_data_2})

                difference_data = y_data - predict_data

                #print(y_data)
                #print(predict_data)
                print(type(y_data))
                print(type(difference_data))
                # 조건부 리스트
                print('차이가 ', difference, ' 이상인 데이터')

                bool_difference_data = difference_data[:] > difference
                print(data_time[bool_difference_data])

                print('상관관계',correlation.correlation(y_data.tolist(), predict_data.tolist()))

                #print('Difference Data:', difference_data)
                print('Max:', np.max(y_data - predict_data))
                print('Predict MSE Cost:', predict_cost_mse)
                print('Predict ABS Cost:', predict_cost_abs)


                print('ckpt name : ', ckpt_name)
                print('datetime : ', datetime.datetime.now())
                print('epochs : ', epochs)
                print('Total time : %f' % (time.time() - start))

                sess.close()
            except Exception as ex:
                print('Exception sess run : ', ex)
                pass
            finally:
                sess.close()

    except Exception as ex:
        print("train_exception")
        traceback.print_exc()
        pass


