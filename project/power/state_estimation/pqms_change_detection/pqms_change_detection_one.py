


# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import datetime


import statsmodels.api as sm

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
    dir_output = 'C:\\_data\\pqms_load_decompose_plot_one\\' + str(period) + '\\'
    colors = ['b', 'o', 'r']

    if not os.path.isdir(dir_output):
        os.makedirs(dir_output)

    filepaths = ['C:\_data\부하데이터\수원경기본부\남시화변전소\ID20\남시화_3상.xls'
                , 'C:\_data\부하데이터\수원경기본부\금촌변전소\ID5\금촌_3상.xls']


    for filepath in filepaths:

        dirname = os.path.dirname(filepath)
        filename = os.path.basename(filepath)


        df_temp = pd.read_excel(filepath)
        df_temp = df_temp.rename(columns={'Unnamed: 1': 'load'})


        bool_possible = is_possible(df_temp, 'load')
        if bool_possible and filename.find('3상') > -1: #
            df_temp.set_index(pd.to_datetime(df_temp[df_temp.columns[0]]), inplace=True)
            df_H_max = df_temp.resample('H').max()
            df_H_mean = df_temp.resample('H').mean()
            df_H_min = df_temp.resample('H').min()



            # 주별 통계량 max
            df_H_max['datetime'] = df_H_max.index.values
            df_H_max['weekday'] = df_H_max['datetime'].dt.weekday
            df_H_max['day_of_year'] = df_H_max['datetime'].dt.dayofyear
            df_H_max['day_of_years'] = df_H_max['datetime'].apply(lambda x: df_H_max.loc[x, 'day_of_year'] + sum([datetime.date(item, 12, 31).timetuple().tm_yday for item in list(set(df_H_max['datetime'].dt.year.values)) if item < df_H_max.loc[x, 'datetime'].year]))


            df_H_max['date_group_week'] = df_H_max['datetime'].apply(lambda x: df_H_max.loc[x, 'day_of_years'] - (df_H_max.loc[x, 'weekday'] + 1))
            df_H_max = df_H_max[(df_H_max['date_group_week'] != df_H_max['date_group_week'].min()) & (df_H_max['date_group_week'] != df_H_max['date_group_week'].max())]

            df_H_max['data_week_odd'] = df_H_max.loc[df_H_max['date_group_week'] % 2 == 0, 'load']
            df_H_max['data_week_even'] = df_H_max.loc[df_H_max['date_group_week'] % 2 == 1, 'load']


            df_H_mean_des_week = pd.DataFrame(df_H_max.groupby('date_group_week')['load'].describe())
            df_H_mean_des_week['date_group_week'] = df_H_mean_des_week.index
            df_H_mean_des_week.rename(columns={'count': 'week_count', 'mean': 'week_mean', 'std': 'week_std', 'min': 'week_min', '25%': 'week_25%', '50%': 'week_50%', '75%': 'week_75%', 'max': 'week_max', 'yyyymmdd': 'yyyymmdd'}, inplace=True)
            df_H_max = pd.merge(df_H_max, df_H_mean_des_week, on='date_group_week', how='left')


            df_H_max['diff_week_max'] = df_H_max['week_max'].diff()
            df_H_max['diff_week_mean'] = df_H_max['week_mean'].diff()
            df_H_max['diff_week_min'] = df_H_max['week_min'].diff()



            # 주별 통계량 mean
            df_H_mean['datetime'] = df_H_mean.index.values
            df_H_mean['weekday'] = df_H_mean['datetime'].dt.weekday
            df_H_mean['day_of_year'] = df_H_mean['datetime'].dt.dayofyear
            df_H_mean['day_of_years'] = df_H_mean['datetime'].apply(lambda x: df_H_mean.loc[x, 'day_of_year'] + sum([datetime.date(item, 12, 31).timetuple().tm_yday for item in list(set(df_H_mean['datetime'].dt.year.values)) if item < df_H_mean.loc[x, 'datetime'].year]))


            df_H_mean['date_group_week'] = df_H_mean['datetime'].apply(lambda x: df_H_mean.loc[x, 'day_of_years'] - (df_H_mean.loc[x, 'weekday'] + 1))
            df_H_mean = df_H_mean[(df_H_mean['date_group_week'] != df_H_mean['date_group_week'].min()) & (df_H_mean['date_group_week'] != df_H_mean['date_group_week'].max())]

            df_H_mean['data_week_odd'] = df_H_mean.loc[df_H_mean['date_group_week'] % 2 == 0, 'load']
            df_H_mean['data_week_even'] = df_H_mean.loc[df_H_mean['date_group_week'] % 2 == 1, 'load']


            df_H_mean_des_week = pd.DataFrame(df_H_mean.groupby('date_group_week')['load'].describe())
            df_H_mean_des_week['date_group_week'] = df_H_mean_des_week.index
            df_H_mean_des_week.rename(columns={'count': 'week_count', 'mean': 'week_mean', 'std': 'week_std', 'min': 'week_min', '25%': 'week_25%', '50%': 'week_50%', '75%': 'week_75%', 'max': 'week_max', 'yyyymmdd': 'yyyymmdd'}, inplace=True)
            df_H_mean = pd.merge(df_H_mean, df_H_mean_des_week, on='date_group_week', how='left')


            df_H_mean['diff_week_max'] = df_H_mean['week_max'].diff()
            df_H_mean['diff_week_mean'] = df_H_mean['week_mean'].diff()
            df_H_mean['diff_week_min'] = df_H_mean['week_min'].diff()



            # 주별 통계량 min
            df_H_min['datetime'] = df_H_min.index.values
            df_H_min['weekday'] = df_H_min['datetime'].dt.weekday
            df_H_min['day_of_year'] = df_H_min['datetime'].dt.dayofyear
            df_H_min['day_of_years'] = df_H_min['datetime'].apply(lambda x: df_H_min.loc[x, 'day_of_year'] + sum([datetime.date(item, 12, 31).timetuple().tm_yday for item in list(set(df_H_min['datetime'].dt.year.values)) if item < df_H_min.loc[x, 'datetime'].year]))


            df_H_min['date_group_week'] = df_H_min['datetime'].apply(lambda x: df_H_min.loc[x, 'day_of_years'] - (df_H_min.loc[x, 'weekday'] + 1))
            df_H_min = df_H_min[(df_H_min['date_group_week'] != df_H_min['date_group_week'].min()) & (df_H_min['date_group_week'] != df_H_min['date_group_week'].max())]

            df_H_min['data_week_odd'] = df_H_min.loc[df_H_min['date_group_week'] % 2 == 0, 'load']
            df_H_min['data_week_even'] = df_H_min.loc[df_H_min['date_group_week'] % 2 == 1, 'load']


            df_H_mean_des_week = pd.DataFrame(df_H_min.groupby('date_group_week')['load'].describe())
            df_H_mean_des_week['date_group_week'] = df_H_mean_des_week.index
            df_H_mean_des_week.rename(columns={'count': 'week_count', 'mean': 'week_mean', 'std': 'week_std', 'min': 'week_min', '25%': 'week_25%', '50%': 'week_50%', '75%': 'week_75%', 'max': 'week_max', 'yyyymmdd': 'yyyymmdd'}, inplace=True)
            df_H_min = pd.merge(df_H_min, df_H_mean_des_week, on='date_group_week', how='left')


            df_H_min['diff_week_max'] = df_H_min['week_max'].diff()
            df_H_min['diff_week_mean'] = df_H_min['week_mean'].diff()
            df_H_min['diff_week_min'] = df_H_min['week_min'].diff()



            # 일별 통계량
            df_H_mean['yyyymmdd'] = df_H_mean['datetime'].dt.strftime('%Y%m%d')
            df_data_decom_des_day = pd.DataFrame(df_H_mean.groupby('yyyymmdd')['load'].describe())
            df_data_decom_des_day['yyyymmdd'] = df_data_decom_des_day.index
            df_data_decom_des_day.rename(columns={'count': 'day_count', 'mean': 'day_mean', 'std': 'day_std', 'min': 'day_min', '25%': 'day_25%', '50%': 'day_50%', '75%': 'day_75%', 'max': 'day_max', 'yyyymmdd': 'yyyymmdd'}, inplace=True)
            df_H_mean = pd.merge(df_H_mean, df_data_decom_des_day, on='yyyymmdd', how='left')

            df_H_mean['diff_day_max'] = df_H_mean['day_max'].diff()
            df_H_mean['diff_day_mean'] = df_H_mean['day_mean'].diff()
            df_H_mean['diff_day_min'] = df_H_mean['day_min'].diff()


            df_H_max.set_index(df_H_max['datetime'], inplace=True)
            df_H_mean.set_index(df_H_mean['datetime'], inplace=True)
            df_H_min.set_index(df_H_min['datetime'], inplace=True)


            df_H_max['date'] = df_H_max.index.to_series().dt.strftime('%Y-%m-%d').astype(str)
            df_H_max.reindex([idx for idx in range(len(df_H_max))])


            df_H_mean['date'] = df_H_mean.index.to_series().dt.strftime('%Y-%m-%d').astype(str)
            df_H_mean.reindex([idx for idx in range(len(df_H_mean))])


            df_H_min['date'] = df_H_min.index.to_series().dt.strftime('%Y-%m-%d').astype(str)
            df_H_min.reindex([idx for idx in range(len(df_H_min))])


            # 일별 부하 데이터의 범위가 임계치를 초과하여 변동한 시점 탐지
            df_range_day = df_H_mean.resample('D')['load'].max() - df_H_mean.resample('D')['load'].min()


            fig = plt.figure(figsize=(35,14))
            gs = gridspec.GridSpec(4, 5)
            ax0_0 = plt.subplot(gs[0, 0])
            ax1_0 = plt.subplot(gs[1, 0], sharex=ax0_0)
            ax2_0 = plt.subplot(gs[2, 0], sharex=ax0_0)
            ax3_0 = plt.subplot(gs[3, 0], sharex=ax0_0)


            ax0_1 = plt.subplot(gs[0, 1])
            ax1_1 = plt.subplot(gs[1, 1], sharex=ax0_0)


            ax2_1 = plt.subplot(gs[2, 1], sharex=ax0_0)
            ax3_1 = plt.subplot(gs[3, 1])


            ax0_3 = plt.subplot(gs[0, 3], sharex=ax0_0)
            ax1_3 = plt.subplot(gs[1, 3], sharex=ax0_0)
            ax2_3 = plt.subplot(gs[2, 3], sharex=ax0_0)
            ax3_3 = plt.subplot(gs[3, 3], sharex=ax0_0)

            ax0_4 = plt.subplot(gs[0, 4], sharex=ax0_0)
            ax1_4 = plt.subplot(gs[1, 4], sharex=ax0_0)
            ax2_4 = plt.subplot(gs[2, 4], sharex=ax0_0)
            ax3_4 = plt.subplot(gs[3, 4], sharex=ax0_0)


            xticklabels = df_H_mean.index[::168].strftime('%Y-%m-%d %H:%M:%s')
            xlim_f = df_H_mean.index[0] + datetime.timedelta(days=-7)
            xlim_b = df_H_mean.index[-1] + datetime.timedelta(days=7)



            # ax2_1.plot(df_data_decom['weekday'] * 100, label='weekday')
            ax1_1.plot(df_H_mean['data_week_odd'], 'b')
            ax1_1.plot(df_H_mean['data_week_even'], 'r')
            ax1_1.plot(df_H_max['week_max'])
            ax1_1.plot(df_H_mean['week_mean'])
            ax1_1.plot(df_H_min['week_min'])



            ax2_1.plot(df_H_mean['day_max'], label='day_max')
            ax2_1.plot(df_H_mean['day_mean'], label='day_mean')
            ax2_1.plot(df_H_mean['day_min'], label='day_min')


            ax3_1.hist(df_H_mean['day_max'].dropna())


            ax0_3.plot(df_H_mean['diff_day_max'], label='diff_day_max')
            ax1_3.plot(df_H_mean['diff_day_mean'], label='diff_day_mean')
            ax2_3.plot(df_H_mean['diff_day_min'], label='diff_day_min')


            ax0_4.plot(df_H_mean['diff_week_max'], label='diff_week_max')
            ax1_4.plot(df_H_mean['diff_week_mean'], label='diff_week_mean')
            ax2_4.plot(df_H_mean['diff_week_min'], label='diff_week_min')


            # ax0_4.plot(df_4H_mean['diff_load'])
            #
            # ax1_4.plot(df_4H_mean['diff_reg_load'])
            # ax1_4.set_ylim(0, 1)
            #
            # ax2_4.plot(df_4H_mean['mean_change_detection'])


            for ax in fig.axes:
                plt.sca(ax)
                plt.xticks(rotation=30)
                plt.legend()

            plt.legend()
            plt.savefig(dir_output + filepath.replace('C:\\_data\\부하데이터\\', '').replace('\\', '_').replace('.xls', '.png'))
            plt.show()
            plt.close()


if __name__ == '__main__':
    period = 168
    plot_pqms_decomposition(period)


