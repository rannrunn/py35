



# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import statsmodels.api as sm
from stldecompose.forecast_funcs import drift
from stldecompose import decompose, forecast

from multiprocessing import Pool


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





def plot_pqms_decomposition(period):
    dir_source = 'C:\\_data\\부하데이터\\'
    dir_output = 'C:\\_data\\pqms_sm_decompose_plot_all\\' + str(period) + '\\'

    if not os.path.isdir(dir_output):
        os.makedirs(dir_output)


    for path, _, filenames in os.walk(dir_source):
        for filename in filenames:

            filepath = os.path.join(path, filename)


            df_temp = pd.read_excel(filepath)
            df_temp = df_temp.rename(columns={'Unnamed: 1': 'load'})

            bool_possible = is_possible(df_temp, 'load')
            if bool_possible:
                df_temp.set_index(pd.to_datetime(df_temp[df_temp.columns[0]]), inplace=True)
                df_temp = df_temp.resample('D').mean().interpolate('linear')
                df_temp = sm.tsa.seasonal_decompose(df_temp['load'], freq=period)

                df_temp.plot()
                plt.xticks(rotation=30)
                plt.legend()
                plt.savefig(dir_output + path.replace(dir_source, '').replace('\\', '_') + filename.replace('.xls', '.png'))
                plt.show()
                plt.close()





if __name__ == '__main__':
    period = 7
    plot_pqms_decomposition(period)


