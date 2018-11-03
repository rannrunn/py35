# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def is_possible(data, col_name='load'):
    # todo: code 정리
    # --- 현빈 ver --- #
    if col_name not in data.columns:
        data.rename(columns={data.columns[0]: 'time', data.columns[1]: col_name}, inplace=True)
        data = data[['time', col_name]]

    # todo: abnormal rate 왜 이렇게 높게 나오지 ... ?

    num_nan = len(data[col_name][np.isnan(data[col_name])])
    num_successive_same_val = len(data[col_name][data[col_name] > 0]) - len(data[col_name][data[col_name] > 0].unique())
    num_under_0 = len(data[data[col_name] <= 0])
    num_of_abnormal_data =  num_nan + num_successive_same_val + num_under_0
    rate_of_abnormal_data = num_of_abnormal_data / len(data[col_name])

    # ///
    # pre.remove_abnormal_from_pd_series(data)
    # ///
    if rate_of_abnormal_data >= 0.3:    # 원래 0.3
        is_possible_data = False
    else:

        recent_data = data[col_name][-6*24*7:]        # data point 1008 (6*24*7) equals to 7days
        num_of_abnormal_data = len(recent_data[np.isnan(recent_data)]) + len(recent_data[recent_data > 0]) \
                               - len(recent_data[recent_data > 0].unique()) + len(recent_data[recent_data <= 0])
        recent_abnormal_rate = num_of_abnormal_data/len(recent_data)

        if recent_abnormal_rate >= 0.5:
            is_possible_data = False
        else:
            is_possible_data = True

    return is_possible_data

dir_source = 'C:\\_data\\부하데이터'
dir_output = 'C:\\_data\\pqms_load_plot\\'

if not os.path.isdir(dir_output):
    os.makedirs(dir_output)

cnt = 0
for root, dirs, files in os.walk(dir_source):
    for file in files:
        if file.endswith(".xls") and file.find('3상') > -1:
            print(os.path.join(root, file))
            df_temp = pd.read_excel(os.path.join(root, file))
            df_temp = df_temp.rename(columns={'Unnamed: 1': 'load'})

            bool_possible = is_possible(df_temp, 'load')
            if bool_possible:
                plt.plot(df_temp.iloc[:, 0], df_temp.iloc[:, 1])
                plt.xticks(rotation=30)
                plt.title(df_temp.columns[0])
                plt.legend()
                plt.savefig(dir_output + root.replace(dir_source + '\\', '').replace('\\', '_') + file.split('.')[0] + '.png', bbox_inches='tight')
                plt.close()

            cnt += 1







