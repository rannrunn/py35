


# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import datetime

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
    dir_output = 'C:\\_data\\pqms_change_detection_all_minmax\\' + str(period) + '\\'

    if not os.path.isdir(dir_output):
        os.makedirs(dir_output)


    for path, dirs, filenames in os.walk(dir_source):

        for filename in filenames:

            filepath = os.path.join(path, filename)

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
                th_reg_resid = np.abs(df_data_decom['reg_resid'].mean()) + (df_data_decom['reg_resid'].std() * 1.5)
                df_data_decom['diff_observed'] = df_data_decom['observed'].diff().abs()
                df_data_decom['diff_reg_resid'] = df_data_decom['reg_resid'].diff().abs()
                df_data_decom['resid_change_detection'] = 0
                df_data_decom.loc[(df_data_decom['reg_resid'].abs() > th_reg_resid), 'resid_change_detection'] = 1

                df_data_decom['datetime'] = df_data_decom.index.values
                df_data_decom['weekday'] = df_data_decom['datetime'].dt.weekday
                df_data_decom['day_of_year'] = df_data_decom['datetime'].dt.dayofyear
                df_data_decom['day_of_years'] = df_data_decom['datetime'].apply(lambda x: df_data_decom.loc[x, 'day_of_year'] + sum([datetime.date(item, 12, 31).timetuple().tm_yday for item in list(set(df_data_decom['datetime'].dt.year.values)) if item < df_data_decom.loc[x, 'datetime'].year]))


                df_data_decom['date_group'] = df_data_decom['datetime'].apply(lambda x: df_data_decom.loc[x, 'day_of_years'] - (df_data_decom.loc[x, 'weekday'] + 1))
                df_data_decom = df_data_decom[(df_data_decom['date_group'] != df_data_decom['date_group'].min()) & (df_data_decom['date_group'] != df_data_decom['date_group'].max())]

                df_data_decom['data_week_odd'] = df_data_decom.loc[df_data_decom['date_group'] % 2 == 0, 'observed']
                df_data_decom['data_week_even'] = df_data_decom.loc[df_data_decom['date_group'] % 2 == 1, 'observed']


                df_data_decom_des = pd.DataFrame(df_data_decom.groupby('date_group')['observed'].describe())
                df_data_decom_des['date_group'] = df_data_decom_des.index
                df_data_decom = pd.merge(df_data_decom, df_data_decom_des, on='date_group', how='left')

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


                # # STL Decomposition Data Plot
                # plt.plot(df_data_decom['observed'], label='observed')
                # plt.plot(df_data_decom['trend'], label='trend')
                # plt.plot(df_data_decom['detrend'], label='detrend')
                # plt.plot(df_data_decom['seasonal'], label='seasonal')
                # plt.plot(df_data_decom['resid'], label='resid')
                # plt.legend()
                # plt.show()
                # plt.close()
                #
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
                df_4H_mean['diff_load'] = df_4H_mean['load'].diff().abs()
                df_4H_mean['diff_reg_load'] = df_4H_mean['reg_load'].diff().abs()
                df_4H_mean['mean_change_detection'] = 0
                # 4시간 부하 데이터가 이전 시점보다 특정 비율을 초과하여 변동한 시점 탐지
                df_4H_mean.loc[(df_4H_mean['diff_reg_load'].abs() > 0.3) & (df_4H_mean['diff_load'].abs() > 2000), 'mean_change_detection'] = 1


                # 일별 부하 데이터의 범위가 임계치를 초과하여 변동한 시점 탐지
                df_range_day = df_4H_mean.resample('D')['load'].max() - df_4H_mean.resample('D')['load'].min()


                # 일별 부하 데이터의 최대값이 임계치를 초과하여 변동한 시점 탐지
                # 최소값은 거의 일정하므로 최대값만으로 탐지하면 될 것 같음
                df_day_max = regularization(df_day_max, 'load', 'reg_load')
                df_day_max['diff_load'] = df_day_max['load'].diff().abs()
                df_day_max['diff_reg_load'] = df_day_max['reg_load'].diff().abs()
                df_day_max['max_change_detection'] = 0
                df_day_max.loc[(df_day_max['diff_reg_load'].abs() > 0.2) & (df_day_max['diff_load'].abs() > 2000), 'max_change_detection'] = 1


                df_4H_mean['date'] = df_4H_mean.index.to_series().dt.strftime('%Y-%m-%d').astype(str)
                df_4H_mean.reindex([idx for idx in range(len(df_4H_mean))])


                fig = plt.figure(figsize=(22,10))
                gs = gridspec.GridSpec(4, 3)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0:2, 1])
                ax3 = fig.add_subplot(4,3,3)
                ax4 = fig.add_subplot(4,3,4)


                ax6 = fig.add_subplot(4,3,6)
                ax7 = fig.add_subplot(4,3,7)
                ax8 = fig.add_subplot(gs[2:4, 1])
                ax9 = fig.add_subplot(4,3,9)
                ax10 = fig.add_subplot(4,3,10)
                ax12 = fig.add_subplot(4,3,12)


                ax1.plot(df_4H_mean['load'], label='observed')
                ax1.plot(data_decom.trend, label='trend')
                # ax1.set_xticklabels(df_4H_mean['date'], rotation=30)
                ax1.set_xticks(df_data_decom.index[::168])
                ax1.set_xticklabels(df_data_decom.index[::168], rotation=30)
                ax1.legend()

                ax4.plot(df_4H_mean['load'] - data_decom.trend, label='detrend')
                # ax4.set_xticklabels(df_4H_mean['date'], rotation=30)
                ax4.set_xticks(df_data_decom.index[::168])
                ax4.set_xticklabels(df_data_decom.index[::168], rotation=30)
                ax4.legend()

                ax7.plot(data_decom.seasonal, label='seasonal')
                # ax7.set_xticklabels(df_4H_mean['date'], rotation=30)
                ax7.set_xticks(df_data_decom.index[::168])
                ax7.set_xticklabels(df_data_decom.index[::168], rotation=30)
                ax7.legend()

                ax10.plot(data_decom.resid.abs(), label='resid')
                # ax10.set_xticklabels(df_4H_mean['date'], rotation=30)
                ax10.set_xticks(df_data_decom.index[::168])
                ax10.set_xticklabels(df_data_decom.index[::168], rotation=30)
                ax10.legend()



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


                ax2.plot(df_data_decom['resid_change_detection'], label='resid_change_detection')
                # ax2.set_xticklabels(df_4H_mean['date'], rotation=30)
                ax2.set_xticks(df_data_decom.index[::168])
                ax2.set_xticklabels(df_data_decom.index[::168], rotation=30)
                ax2.legend()


                # ax5.plot(df_data_decom['reg_resid'], label='reg_resid')
                # ax5.set_ylim(0, 1)
                # # ax5.set_xticklabels(df_4H_mean['date'], rotation=30)
                # ax5.legend()


                # ax8.plot(df_data_decom['weekday'] * 100, label='weekday')
                ax8.plot(df_data_decom['data_week_odd'], 'b', label='data_week_odd')
                ax8.plot(df_data_decom['data_week_even'], 'r', label='data_week_even')
                ax8.plot(df_data_decom['max'], label='max')
                ax8.plot(df_data_decom['min'], label='min')
                ax8.plot(df_data_decom['mean'], label='mean')
                ax8.plot(df_data_decom['std'], label='std')
                ax8.set_xticks(df_data_decom.index[::168])
                ax8.set_xticklabels(df_data_decom.index[::168], rotation=30)
                ax8.legend()



                ax3.plot(df_4H_mean['diff_load'])
                ax3.set_ylabel('4H Difference Load Data')
                # ax3.set_xticklabels(df_4H_mean['date'], rotation=30)
                ax3.set_xticks(df_data_decom.index[::168])
                ax3.set_xticklabels(df_data_decom.index[::168], rotation=30)
                ax3.legend()

                ax6.plot(df_4H_mean['diff_reg_load'])
                ax6.set_ylabel('4H Difference Regular Load Data')
                ax6.set_ylim(0, 1)
                # ax6.set_xticklabels(df_4H_mean['date'], rotation=30)
                ax6.set_xticks(df_data_decom.index[::168])
                ax6.set_xticklabels(df_data_decom.index[::168], rotation=30)
                ax6.legend()

                ax9.plot(df_4H_mean['mean_change_detection'])
                ax9.set_ylabel('4H Mean Change Detection')
                # ax9.set_xticklabels(df_4H_mean['date'], rotation=30)
                ax9.set_xticks(df_data_decom.index[::168])
                ax9.set_xticklabels(df_data_decom.index[::168], rotation=30)
                ax9.legend()



                plt.legend()
                # plt.savefig(dir_output + filepath.replace('C:\\_data\\부하데이터\\', '').replace('\\', '_').replace('.xls', '.png'))
                plt.show()
                plt.close()


if __name__ == '__main__':
    period = 42
    plot_pqms_decomposition(period)


