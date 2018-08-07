


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

    # filepath = 'C:\_data\부하데이터\수원경기본부\남시화변전소\ID20\남시화_3상.xls'
    filepath = 'C:\_data\부하데이터\수원경기본부\금촌변전소\ID5\금촌_3상.xls'

    dirname = os.path.dirname(filepath)
    filename = os.path.basename(filepath)


    df_temp = pd.read_excel(filepath)
    df_temp = df_temp.rename(columns={'Unnamed: 1': 'load'})


    bool_possible = is_possible(df_temp, 'load')
    if bool_possible and filename.find('3상') > -1: #
        df_temp.set_index(pd.to_datetime(df_temp[df_temp.columns[0]]), inplace=True)
        df_4H_mean = df_temp.resample('4H').mean().interpolate('linear')
        df_day_max = df_temp.resample('D').max()
        df_day_min = df_temp.resample('D').min()
        data_decom = sm.tsa.seasonal_decompose(df_4H_mean['load'], freq=period)

        df_data_decom = pd.DataFrame({"observed":data_decom.observed})
        df_data_decom['trend'] = data_decom.trend
        df_data_decom['detrend'] = df_data_decom['observed'] - df_data_decom['trend']
        df_data_decom['seasonal'] = data_decom.seasonal
        df_data_decom['resid'] = data_decom.resid

        df_data_decom = regularization(df_data_decom, 'resid', 'reg_resid')
        th_reg_resid = df_data_decom['reg_resid'].mean() + (df_data_decom['reg_resid'].std() * 1.5)
        df_data_decom['diff_observed'] = df_data_decom['observed'].diff()
        df_data_decom['diff_reg_resid'] = df_data_decom['reg_resid'].diff()
        df_data_decom['resid_change_detection'] = 0
        df_data_decom.loc[(df_data_decom['reg_resid'].abs() > th_reg_resid), 'resid_change_detection'] = 1


        # 주별 통계량
        df_data_decom['datetime'] = df_data_decom.index.values
        df_data_decom['weekday'] = df_data_decom['datetime'].dt.weekday
        df_data_decom['day_of_year'] = df_data_decom['datetime'].dt.dayofyear
        df_data_decom['day_of_years'] = df_data_decom['datetime'].apply(lambda x: df_data_decom.loc[x, 'day_of_year'] + sum([datetime.date(item, 12, 31).timetuple().tm_yday for item in list(set(df_data_decom['datetime'].dt.year.values)) if item < df_data_decom.loc[x, 'datetime'].year]))


        df_data_decom['date_group_week'] = df_data_decom['datetime'].apply(lambda x: df_data_decom.loc[x, 'day_of_years'] - (df_data_decom.loc[x, 'weekday'] + 1))
        df_data_decom = df_data_decom[(df_data_decom['date_group_week'] != df_data_decom['date_group_week'].min()) & (df_data_decom['date_group_week'] != df_data_decom['date_group_week'].max())]

        df_data_decom['data_week_odd'] = df_data_decom.loc[df_data_decom['date_group_week'] % 2 == 0, 'observed']
        df_data_decom['data_week_even'] = df_data_decom.loc[df_data_decom['date_group_week'] % 2 == 1, 'observed']


        df_data_decom_des_week = pd.DataFrame(df_data_decom.groupby('date_group_week')['observed'].describe())
        df_data_decom_des_week['date_group_week'] = df_data_decom_des_week.index
        df_data_decom_des_week.rename(columns={'count': 'week_count', 'mean': 'week_mean', 'std': 'week_std', 'min': 'week_min', '25%': 'week_25%', '50%': 'week_50%', '75%': 'week_75%', 'max': 'week_max', 'yyyymmdd': 'yyyymmdd'}, inplace=True)
        df_data_decom = pd.merge(df_data_decom, df_data_decom_des_week, on='date_group_week', how='left')


        df_data_decom['diff_week_max'] = df_data_decom['week_max'].diff()
        df_data_decom['diff_week_mean'] = df_data_decom['week_mean'].diff()
        df_data_decom['diff_week_min'] = df_data_decom['week_min'].diff()


        # 일별 통계량
        df_data_decom['yyyymmdd'] = df_data_decom['datetime'].dt.strftime('%Y%m%d')
        df_data_decom_des_day = pd.DataFrame(df_data_decom.groupby('yyyymmdd')['observed'].describe())
        df_data_decom_des_day['yyyymmdd'] = df_data_decom_des_day.index
        df_data_decom_des_day.rename(columns={'count': 'day_count', 'mean': 'day_mean', 'std': 'day_std', 'min': 'day_min', '25%': 'day_25%', '50%': 'day_50%', '75%': 'day_75%', 'max': 'day_max', 'yyyymmdd': 'yyyymmdd'}, inplace=True)
        df_data_decom = pd.merge(df_data_decom, df_data_decom_des_day, on='yyyymmdd', how='left')

        df_data_decom['diff_day_max'] = df_data_decom['day_max'].diff()
        df_data_decom['diff_day_mean'] = df_data_decom['day_mean'].diff()
        df_data_decom['diff_day_min'] = df_data_decom['day_min'].diff()

        
        df_data_decom.set_index(df_data_decom['datetime'], inplace=True)

        # plt.plot(df_data_decom['observed'])
        # plt.plot(df_data_decom['weekday'] * 100)
        # plt.plot(df_data_decom['data_week_odd'], 'b')
        # plt.plot(df_data_decom['data_week_even'], 'r')
        # plt.plot(df_data_decom['max'], label='max')
        # plt.plot(df_data_decom['min'], label='min')
        # plt.plot(df_data_decom['mean'], label='mean')
        # plt.plot(df_data_decom['std'], label='std')
        # plt.legend()
        # plt.show()
        # plt.close()

        # # 정규화 데이터 플롯
        # plt.plot(df_data_decom['reg_resid'], label='reg_resid')
        # plt.legend()
        # plt.show()
        # plt.close()
        #
        # # 정규화 데이터 플롯
        # plt.plot(df_data_decom['resid_change_detection'], label='resid_change_detection')
        # plt.legend()
        # plt.show()
        # plt.close()


        df_4H_mean = regularization(df_4H_mean, 'load', 'reg_load')
        df_4H_mean['diff_load'] = df_4H_mean['load'].diff()
        df_4H_mean['diff_reg_load'] = df_4H_mean['reg_load'].diff()
        df_4H_mean['mean_change_detection'] = 0
        # 4시간 부하 데이터가 이전 시점보다 특정 비율을 초과하여 변동한 시점 탐지
        df_4H_mean.loc[(df_4H_mean['diff_reg_load'].abs() > 0.3) & (df_4H_mean['diff_load'].abs() > 2000), 'mean_change_detection'] = 1


        # 일별 부하 데이터의 범위가 임계치를 초과하여 변동한 시점 탐지
        df_range_day = df_4H_mean.resample('D')['load'].max() - df_4H_mean.resample('D')['load'].min()


        # 일별 부하 데이터의 최대값이 임계치를 초과하여 변동한 시점 탐지
        # 최소값은 거의 일정하므로 최대값만으로 탐지하면 될 것 같음
        df_day_max = regularization(df_day_max, 'load', 'reg_load')
        df_day_max['diff_load'] = df_day_max['load'].diff()
        df_day_max['diff_reg_load'] = df_day_max['reg_load'].diff()
        df_day_max['max_change_detection'] = 0
        df_day_max.loc[(df_day_max['diff_reg_load'].abs() > 0.2) & (df_day_max['diff_load'].abs() > 2000), 'max_change_detection'] = 1


        df_4H_mean['date'] = df_4H_mean.index.to_series().dt.strftime('%Y-%m-%d').astype(str)
        df_4H_mean.reindex([idx for idx in range(len(df_4H_mean))])


        fig = plt.figure(figsize=(35,14))
        gs = gridspec.GridSpec(4, 5)
        ax0_0 = plt.subplot(gs[0, 0])
        ax1_0 = plt.subplot(gs[1, 0], sharex=ax0_0)
        ax2_0 = plt.subplot(gs[2, 0], sharex=ax0_0)
        ax3_0 = plt.subplot(gs[3, 0], sharex=ax0_0)


        ax0_1 = plt.subplot(gs[0:2, 1])
        ax2_1 = plt.subplot(gs[2:4, 1], sharex=ax0_0)


        ax0_2 = plt.subplot(gs[0:2, 2], sharex=ax0_0)
        ax2_2 = plt.subplot(gs[2:4, 2])


        ax0_3 = plt.subplot(gs[0, 3], sharex=ax0_0)
        ax1_3 = plt.subplot(gs[1, 3], sharex=ax0_0)
        ax2_3 = plt.subplot(gs[2, 3], sharex=ax0_0)
        ax3_3 = plt.subplot(gs[3, 3], sharex=ax0_0)

        ax0_4 = plt.subplot(gs[0, 4], sharex=ax0_0)
        ax1_4 = plt.subplot(gs[1, 4], sharex=ax0_0)
        ax2_4 = plt.subplot(gs[2, 4], sharex=ax0_0)
        ax3_4 = plt.subplot(gs[3, 4], sharex=ax0_0)


        xticklabels = df_data_decom.index[::168].strftime('%Y-%m-%d %H:%M:%s')
        xlim_f = df_4H_mean.index[0] + datetime.timedelta(days=-7)
        xlim_b = df_4H_mean.index[-1] + datetime.timedelta(days=7)


        ax0_0.plot(df_4H_mean.index, df_4H_mean['load'], label='observed')
        ax0_0.plot(data_decom.trend, label='trend')
        ax0_0.set_xticks(df_data_decom.index[::168])
        ax0_0.set_xticklabels(xticklabels, rotation=30)
        ax1_0.set_xlim(xlim_f, xlim_b)

        ax1_0.plot(df_4H_mean.index, df_4H_mean['load'] - data_decom.trend, label='detrend')

        ax2_0.plot(data_decom.seasonal, label='seasonal')

        ax3_0.plot(data_decom.resid, label='resid')


        ax0_1.hist(data_decom.resid.dropna())


        # ax2_1.plot(df_data_decom['weekday'] * 100, label='weekday')
        ax2_1.plot(df_data_decom['data_week_odd'], 'b', label='data_week_odd')
        ax2_1.plot(df_data_decom['data_week_even'], 'r', label='data_week_even')
        ax2_1.plot(df_data_decom['week_max'], label='week_max')
        ax2_1.plot(df_data_decom['week_mean'], label='week_mean')
        ax2_1.plot(df_data_decom['week_min'], label='week_min')



        ax0_2.plot(df_data_decom['day_max'], label='day_max')
        ax0_2.plot(df_data_decom['day_mean'], label='day_mean')
        ax0_2.plot(df_data_decom['day_min'], label='day_min')


        ax2_2.hist(df_data_decom['day_max'].dropna())


        ax0_3.plot(df_data_decom['diff_day_max'], label='diff_day_max')
        ax1_3.plot(df_data_decom['diff_day_mean'], label='diff_day_mean')
        ax2_3.plot(df_data_decom['diff_day_min'], label='diff_day_min')


        ax0_4.plot(df_data_decom['diff_week_max'], label='diff_week_max')
        ax1_4.plot(df_data_decom['diff_week_mean'], label='diff_week_mean')
        ax2_4.plot(df_data_decom['diff_week_min'], label='diff_week_min')


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
        # plt.savefig(dir_output + filepath.replace('C:\\_data\\부하데이터\\', '').replace('\\', '_').replace('.xls', '.png'))
        plt.show()
        plt.close()


if __name__ == '__main__':
    period = 42
    plot_pqms_decomposition(period)


