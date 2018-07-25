


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


def regularization(df, col, col_reg):
    df[col_reg] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def plot_pqms_decomposition(period):
    dir_source = 'C:\\_data\\부하데이터'
    dir_output = 'C:\\_data\\pqms_change_detection_all\\' + str(period) + '\\'

    if not os.path.isdir(dir_output):
        os.makedirs(dir_output)


    for path, dirs, filenames in os.walk(dir_source):

        for filename in filenames:

            filepath = os.path.join(path, filename)

            df_temp = pd.read_excel(filepath)
            df_temp = df_temp.rename(columns={'Unnamed: 1': 'load'})

            bool_possible = is_possible(df_temp, 'load')
            if bool_possible and filename.find('3상') > -1:
                df_temp.set_index(pd.to_datetime(df_temp[df_temp.columns[0]]), inplace=True)
                df_4H_mean = df_temp.resample('4H').mean().interpolate('linear')
                df_day_max = df_temp.resample('D').max()
                df_day_min = df_temp.resample('D').min()
                data_decom = sm.tsa.seasonal_decompose(df_4H_mean['load'], freq=period)

                df_4H_mean = regularization(df_4H_mean, 'load', 'reg')
                df_4H_mean['diff_load'] = df_4H_mean['load'].diff().abs()
                df_4H_mean['diff_reg_load'] = df_4H_mean['reg'].diff().abs()
                df_4H_mean['mean_change_detection'] = 0
                # 4시간 부하 데이터가 이전 시점보다 특정 비율을 초과하여 변동한 시점 탐지
                df_4H_mean.loc[(df_4H_mean['diff_reg_load'].abs() > 0.3) & (df_4H_mean['diff_load'].abs() > 2000), 'mean_change_detection'] = 1


                # 일별 부하 데이터의 범위가 임계치를 초과하여 변동한 시점 탐지
                df_range_day = df_4H_mean.resample('D')['load'].max() - df_4H_mean.resample('D')['load'].min()


                # 일별 부하 데이터의 최대값이 임계치를 초과하여 변동한 시점 탐지
                # 최소값은 거의 일정하므로 최대값만으로 탐지하면 될 것 같음
                df_day_max = regularization(df_day_max, 'load', 'reg')
                df_day_max['diff_load'] = df_day_max['load'].diff().abs()
                df_day_max['diff_reg_load'] = df_day_max['reg'].diff().abs()
                df_day_max['max_change_detection'] = 0
                df_day_max.loc[(df_day_max['diff_reg_load'].abs() > 0.2) & (df_day_max['diff_load'].abs() > 2000), 'max_change_detection'] = 1


                df_4H_mean['date'] = df_4H_mean.index.to_series().dt.strftime('%Y-%m-%d').astype(str)
                df_4H_mean.reindex([idx for idx in range(len(df_4H_mean))])

                fig = plt.figure(figsize=(22,10))
                ax1 = fig.add_subplot(4,3,1)
                ax2 = fig.add_subplot(4,3,2)
                ax3 = fig.add_subplot(4,3,3)
                ax4 = fig.add_subplot(4,3,4)
                ax5 = fig.add_subplot(4,3,5)
                ax6 = fig.add_subplot(4,3,6)
                ax7 = fig.add_subplot(4,3,7)
                ax8 = fig.add_subplot(4,3,8)
                ax9 = fig.add_subplot(4,3,9)
                ax10 = fig.add_subplot(4,3,10)
                ax11 = fig.add_subplot(4,3,11)
                ax12 = fig.add_subplot(4,3,12)

                ax1.plot(df_4H_mean['load'])
                ax1.plot(data_decom.trend)
                ax1.set_ylabel('Raw Data & Trend')
                # ax1.set_xticklabels(df_4H_mean['date'], rotation=30)

                ax4.plot(df_4H_mean['load'] - data_decom.trend)
                ax4.set_ylabel('Detrend Data')
                # ax4.set_xticklabels(df_4H_mean['date'], rotation=30)

                ax7.plot(data_decom.seasonal)
                ax7.set_ylabel('Seasonal Data')
                # ax7.set_xticklabels(df_4H_mean['date'], rotation=30)

                ax10.plot(data_decom.resid.abs())
                ax10.set_ylabel('Residual Data')
                # ax10.set_xticklabels(df_4H_mean['date'], rotation=30)





                ax2.plot(df_4H_mean['diff_load'])
                ax2.set_ylabel('4H Difference Load Data')
                # ax2.set_xticklabels(df_4H_mean['date'], rotation=30)

                ax5.plot(df_4H_mean['diff_reg_load'])
                ax5.set_ylabel('4H Difference Regular Load Data')
                ax5.set_ylim(0, 1)
                # ax5.set_xticklabels(df_4H_mean['date'], rotation=30)

                ax8.plot(df_4H_mean['mean_change_detection'])
                ax8.set_ylabel('4H Mean Change Detection')
                # ax8.set_xticklabels(df_4H_mean['date'], rotation=30)





                ax3.plot(df_day_max['diff_load'])
                ax3.set_ylabel('Day Difference Load Data')
                # ax3.set_xticklabels(df_4H_mean['date'], rotation=30)

                ax6.plot(df_day_max['diff_reg_load'])
                ax6.set_ylabel('Day Difference Regular Load Data')
                ax6.set_ylim(0, 1)
                # ax6.set_xticklabels(df_4H_mean['date'], rotation=30)

                ax9.plot(df_day_max['max_change_detection'])
                ax9.set_ylabel('Day Max Change Detection')
                # ax9.set_xticklabels(df_4H_mean['date'], rotation=30)

                plt.legend()
                plt.savefig(dir_output + filepath.replace('C:\\_data\\부하데이터\\', '').replace('\\', '_').replace('.xls', '.png'))
                plt.show()
                plt.close()


if __name__ == '__main__':
    period = 42
    plot_pqms_decomposition(period)


