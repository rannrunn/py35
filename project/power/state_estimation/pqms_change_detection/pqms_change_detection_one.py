


# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import datetime
import time


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
        start = time.time()

        dirname = os.path.dirname(filepath)
        filename = os.path.basename(filepath)


        df_temp = pd.read_excel(filepath)
        df_temp = df_temp.rename(columns={'Unnamed: 1': 'load'})


        print('경과시간 1:', (time.time() - start))


        bool_possible = is_possible(df_temp, 'load')
        if bool_possible and filename.find('3상') > -1: #
            df_temp.set_index(pd.to_datetime(df_temp[df_temp.columns[0]]), inplace=True)
            df_H_max = df_temp.resample('4H').max()
            df_H_mean = df_temp.resample('4H').mean()
            df_H_min = df_temp.resample('4H').min()
            df_H_data = pd.DataFrame()


            df_H_data['load_max'] = df_H_max['load']
            df_H_data['load_mean'] = df_H_mean['load']
            df_H_data['load_min'] = df_H_min['load']


            # 주별 통계량 max
            df_H_data['datetime'] = df_H_data.index.values
            df_H_data['weekday'] = df_H_data['datetime'].dt.weekday
            df_H_data['day_of_year'] = df_H_data['datetime'].dt.dayofyear
            df_H_data['day_of_years'] = df_H_data['datetime'].apply(lambda x: df_H_data.loc[x, 'day_of_year'] + sum([datetime.date(item, 12, 31).timetuple().tm_yday for item in list(set(df_H_data['datetime'].dt.year.values)) if item < df_H_data.loc[x, 'datetime'].year]))


            df_H_data['date_group_week'] = df_H_data['datetime'].apply(lambda x: df_H_data.loc[x, 'day_of_years'] - (df_H_data.loc[x, 'weekday'] + 1))
            df_H_data = df_H_data[(df_H_data['date_group_week'] != df_H_data['date_group_week'].min()) & (df_H_data['date_group_week'] != df_H_data['date_group_week'].max())]


            df_H_data['data_week_odd'] = df_H_data.loc[df_H_data['date_group_week'] % 2 == 0, 'load_mean']
            df_H_data['data_week_even'] = df_H_data.loc[df_H_data['date_group_week'] % 2 == 1, 'load_mean']


            df_H_max_des_week = pd.DataFrame(df_H_data.groupby('date_group_week')['load_max'].describe())
            df_H_max_des_week['date_group_week'] = df_H_max_des_week.index
            # df_H_max_des_week.rename(columns={'count': 'week_count', 'mean': 'week_mean', 'std': 'week_std', 'min': 'week_min', '25%': 'week_25%', '50%': 'week_50%', '75%': 'week_75%', 'max': 'week_max', 'yyyymmdd': 'yyyymmdd'}, inplace=True)
            df_H_max_des_week.rename(columns={'max': 'week_max'}, inplace=True)
            df_H_data_des_week = pd.DataFrame({'week_max':df_H_max_des_week['week_max'], 'date_group_week':df_H_max_des_week['date_group_week']}, df_H_max_des_week.index)


            print('경과시간 2:', (time.time() - start))


            # 주별 통계량 mean
            df_H_mean_des_week = pd.DataFrame(df_H_data.groupby('date_group_week')['load_mean'].describe())
            df_H_mean_des_week['date_group_week'] = df_H_mean_des_week.index
            df_H_mean_des_week.rename(columns={'mean': 'week_mean'}, inplace=True)
            df_H_data_des_week['week_mean'] = df_H_mean_des_week['week_mean']


            print('경과시간 3:', (time.time() - start))


            # 주별 통계량 min
            df_H_min_des_week = pd.DataFrame(df_H_data.groupby('date_group_week')['load_min'].describe())
            df_H_min_des_week['date_group_week'] = df_H_min_des_week.index
            df_H_min_des_week.rename(columns={'min': 'week_min'}, inplace=True)
            df_H_data_des_week['week_min'] = df_H_min_des_week['week_min']
            df_H_data_des_week['week_range'] = df_H_data_des_week['week_max'] - df_H_data_des_week['week_min']


            df_H_data = pd.merge(df_H_data, df_H_data_des_week, on='date_group_week', how='left')
            df_H_data_des_week['week_ratio'] = df_H_data_des_week['week_range'] / df_H_data_des_week['week_range'].shift(1)
            df_H_data_des_week['week_change_flag'] = (df_H_data_des_week['week_ratio'] < 0.6) | (df_H_data_des_week['week_ratio'] > 1.4)


            print('경과시간 4:', (time.time() - start))


            df_H_data['diff_week_max'] = df_H_data['week_max'].diff()
            df_H_data['diff_week_mean'] = df_H_data['week_mean'].diff()
            df_H_data['diff_week_min'] = df_H_data['week_min'].diff()


            # 일별 통계량 max
            df_H_data['yyyymmdd'] = df_H_data['datetime'].dt.strftime('%Y%m%d')
            df_H_max_des_day = pd.DataFrame(df_H_data.groupby('yyyymmdd')['load_max'].describe())
            df_H_max_des_day['yyyymmdd'] = df_H_max_des_day.index
            # df_H_max_des_day.rename(columns={'count': 'day_count', 'mean': 'day_mean', 'std': 'day_std', 'min': 'day_min', '25%': 'day_25%', '50%': 'day_50%', '75%': 'day_75%', 'max': 'day_max', 'yyyymmdd': 'yyyymmdd'}, inplace=True)
            df_H_max_des_day.rename(columns={'max': 'day_max'}, inplace=True)
            df_H_data = pd.merge(df_H_data, pd.DataFrame({'yyyymmdd':df_H_max_des_day['yyyymmdd'],'day_max':df_H_max_des_day['day_max']}), on='yyyymmdd', how='left')


            # 일별 통계량 mean
            df_H_data['yyyymmdd'] = df_H_data['datetime'].dt.strftime('%Y%m%d')
            df_H_mean_des_day = pd.DataFrame(df_H_data.groupby('yyyymmdd')['load_mean'].describe())
            df_H_mean_des_day['yyyymmdd'] = df_H_mean_des_day.index
            df_H_mean_des_day.rename(columns={'mean': 'day_mean'}, inplace=True)
            df_H_data = pd.merge(df_H_data, pd.DataFrame({'yyyymmdd':df_H_mean_des_day['yyyymmdd'],'day_mean':df_H_mean_des_day['day_mean']}), on='yyyymmdd', how='left')


            # 일별 통계량 min
            df_H_data['yyyymmdd'] = df_H_data['datetime'].dt.strftime('%Y%m%d')
            df_H_min_des_day = pd.DataFrame(df_H_data.groupby('yyyymmdd')['load_min'].describe())
            df_H_min_des_day['yyyymmdd'] = df_H_min_des_day.index
            df_H_min_des_day.rename(columns={'min': 'day_min'}, inplace=True)
            df_H_data = pd.merge(df_H_data, pd.DataFrame({'yyyymmdd':df_H_min_des_day['yyyymmdd'],'day_min':df_H_min_des_day['day_min']}), on='yyyymmdd', how='left')


            df_H_data['day_range'] = df_H_data['day_max'] - df_H_data['day_min']


            df_H_data['diff_day_max'] = df_H_data['day_max'].diff()
            df_H_data['diff_day_mean'] = df_H_data['day_mean'].diff()
            df_H_data['diff_day_min'] = df_H_data['day_min'].diff()
            df_H_data.set_index(df_H_data['datetime'], inplace=True)


            df_H_data['date'] = df_H_data.index.to_series().dt.strftime('%Y-%m-%d').astype(str)
            df_H_data.reindex([idx for idx in range(len(df_H_data))])


            print('경과시간 5:', (time.time() - start))


            fig = plt.figure(figsize=(35,14))
            gs = gridspec.GridSpec(4, 5)
            ax0_0 = plt.subplot(gs[0, 0])
            ax1_0 = plt.subplot(gs[1, 0])
            ax2_0 = plt.subplot(gs[2, 0], sharex=ax0_0)
            ax3_0 = plt.subplot(gs[3, 0], sharex=ax0_0)


            ax0_1 = plt.subplot(gs[0, 1], sharex=ax0_0)
            ax1_1 = plt.subplot(gs[1, 1], sharex=ax0_0)
            ax2_1 = plt.subplot(gs[2, 1], sharex=ax0_0)
            ax3_1 = plt.subplot(gs[3, 1])


            ax0_2 = plt.subplot(gs[0, 2], sharex=ax0_0)
            ax1_2 = plt.subplot(gs[1, 2], sharex=ax0_0)
            ax2_2 = plt.subplot(gs[2, 2], sharex=ax0_0)
            ax3_2 = plt.subplot(gs[3, 2])


            # ax2_1.plot(df_data_decom['weekday'] * 100, label='weekday')
            ax0_0.plot(df_H_data['data_week_odd'], 'b')
            ax0_0.plot(df_H_data['data_week_even'], 'r')
            ax0_0.plot(df_H_data['week_max'])
            ax0_0.plot(df_H_data['week_mean'])
            ax0_0.plot(df_H_data['week_min'])


            ax1_0.scatter(df_H_data_des_week.index, df_H_data_des_week['week_range'])


            ax2_0.plot(df_H_data['day_max'], label='day_max')
            ax2_0.plot(df_H_data['day_mean'], label='day_mean')
            ax2_0.plot(df_H_data['day_min'], label='day_min')


            ax3_0.scatter(df_H_data.index, df_H_data['day_range'])


            ax0_1.scatter(df_H_data.index, df_H_data['diff_day_max'], s=10, label='diff_day_max')
            ax1_1.scatter(df_H_data.index, df_H_data['diff_day_mean'], s=10, label='diff_day_mean')
            ax2_1.scatter(df_H_data.index, df_H_data['diff_day_min'], s=10, label='diff_day_min')
            ax3_1.scatter(df_H_data_des_week.index, df_H_data_des_week['week_ratio'], label='week_ratio')


            ax0_2.scatter(df_H_data.index, df_H_data['diff_week_max'], s=10, label='diff_week_max')
            ax1_2.scatter(df_H_data.index, df_H_data['diff_week_mean'], s=10, label='diff_week_mean')
            ax2_2.scatter(df_H_data.index, df_H_data['diff_week_min'], s=10, label='diff_week_min')
            ax3_2.scatter(df_H_data_des_week.index, df_H_data_des_week['week_change_flag'], label='week_change_flag')


            for ax in fig.axes:
                plt.sca(ax)
                plt.xticks(rotation=30)
                plt.legend()

            plt.savefig(dir_output + filepath.replace('C:\\_data\\부하데이터\\', '').replace('\\', '_').replace('.xls', '.png'))
            plt.show()
            plt.close()









        print('경과시간 6:', (time.time() - start))


if __name__ == '__main__':
    period = 42
    plot_pqms_decomposition(period)


