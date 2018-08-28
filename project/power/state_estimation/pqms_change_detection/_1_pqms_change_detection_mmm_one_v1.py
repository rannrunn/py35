


# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import datetime
import time

from multiprocessing import Process

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


def detection_change_week(df):

    df['week_flag_change_range'] = False

    load_range_mean = None  # 범위의 평균
    week_num_start = -1
    for week_num in range(df['date_group_week'].min(), df['date_group_week'].max() + 1, 1):

        if len(df.loc[(df['date_group_week'] == week_num), 'week_range']) == 0:
            continue

        load_range_mean = df.loc[(df['date_group_week'] >= week_num_start) & (df['date_group_week'] < week_num - 1), 'week_range'].mean()

        laod_range_ratio = df.loc[(df['date_group_week'] == week_num), 'week_range'].values[0] / load_range_mean

        if laod_range_ratio < 0.714 or laod_range_ratio > 1.4:
            df.loc[(df['date_group_week'] == week_num), 'week_flag_change_range'] = True
            week_num_start = week_num

    return df


def detection_change_day(df):



    return df



def pqms_change_detection(filepath):

    dir_output = 'C:\\_data\\pqms_change_detection_algorithm\\'

    if not os.path.isdir(dir_output):
        os.makedirs(dir_output)

    start = time.time()

    filename = os.path.basename(filepath)
    dirname = os.path.dirname(filepath)


    if not filename.find('3상') > -1: #
        return


    df_temp = pd.read_excel(filepath)
    df_temp = df_temp.rename(columns={'Unnamed: 1': 'load'})


    # 값이 0과 같거나 작을 경우 np.nan으로 대체
    df_temp.loc[df_temp['load'] <= 0, 'load'] = np.nan


    print('경과시간 1:', (time.time() - start))
    bool_possible = is_possible(df_temp, 'load')
    if bool_possible:

        df_temp.set_index(pd.to_datetime(df_temp[df_temp.columns[0]]), inplace=True)

        df_4H_max = df_temp.resample('4H').max()
        df_4H_mean = df_temp.resample('4H').mean()
        df_4H_min = df_temp.resample('4H').min()
        df_4H_data = pd.DataFrame()


        df_4H_data['load_max'] = df_4H_max['load']
        df_4H_data['load_mean'] = df_4H_mean['load']
        df_4H_data['load_min'] = df_4H_min['load']


        # 주별 통계량 max
        df_4H_data['datetime'] = df_4H_data.index.values
        df_4H_data['weekday'] = df_4H_data['datetime'].dt.weekday
        df_4H_data['day_of_year'] = df_4H_data['datetime'].dt.dayofyear
        df_4H_data['day_of_years'] = df_4H_data['datetime'].apply(lambda x: df_4H_data.loc[x, 'day_of_year'] + sum([datetime.date(item, 12, 31).timetuple().tm_yday for item in list(set(df_4H_data['datetime'].dt.year.values)) if item < df_4H_data.loc[x, 'datetime'].year]))


        df_4H_data['date_group_week'] = df_4H_data['datetime'].apply(lambda x: df_4H_data.loc[x, 'day_of_years'] - (df_4H_data.loc[x, 'weekday'] + 1))
        df_4H_data = df_4H_data[(df_4H_data['date_group_week'] != df_4H_data['date_group_week'].min()) & (df_4H_data['date_group_week'] != df_4H_data['date_group_week'].max())]


        df_4H_data['data_week_odd'] = df_4H_data.loc[df_4H_data['date_group_week'] % 2 == 0, 'load_mean']
        df_4H_data['data_week_even'] = df_4H_data.loc[df_4H_data['date_group_week'] % 2 == 1, 'load_mean']


        df_W_des_max = pd.DataFrame(df_4H_data.groupby('date_group_week')['load_max'].describe())
        df_W_des_max['date_group_week'] = df_W_des_max.index
        # df_H_max_des_week.rename(columns={'count': 'week_count', 'mean': 'week_mean', 'std': 'week_std', 'min': 'week_min', '25%': 'week_25%', '50%': 'week_50%', '75%': 'week_75%', 'max': 'week_max', 'yyyymmdd': 'yyyymmdd'}, inplace=True)
        df_W_des_max.rename(columns={'max': 'week_max'}, inplace=True)
        df_W_des = pd.DataFrame({'week_max':df_W_des_max['week_max'], 'date_group_week':df_W_des_max['date_group_week']}, df_W_des_max.index)


        print('경과시간 2:', (time.time() - start))


        # 주별 통계량 mean
        df_W_des_mean = pd.DataFrame(df_4H_data.groupby('date_group_week')['load_mean'].describe())
        df_W_des_mean['date_group_week'] = df_W_des_mean.index
        df_W_des_mean.rename(columns={'mean': 'week_mean', 'std': 'week_std'}, inplace=True)
        df_W_des['week_mean'] = df_W_des_mean['week_mean']
        df_W_des['week_std'] = df_W_des_mean['week_std']


        print('경과시간 3:', (time.time() - start))


        # 주별 통계량 min
        df_W_des_min = pd.DataFrame(df_4H_data.groupby('date_group_week')['load_min'].describe())
        df_W_des_min['date_group_week'] = df_W_des_min.index
        df_W_des_min.rename(columns={'min': 'week_min'}, inplace=True)
        df_W_des['week_min'] = df_W_des_min['week_min']

        # 주단위 범위 계산
        df_W_des['week_range'] = df_W_des['week_max'] - df_W_des['week_min']

        # 주단위 범위 변화 탐지 : 탐지율 높음
        # df_W_des['week_ratio_range'] = df_W_des['week_range'] / df_W_des['week_range'].shift(1)
        # df_W_des['week_flag_change_range'] = (df_W_des['week_ratio_range'] < 0.714) | (df_W_des['week_ratio_range'] > 1.4)

        # 주단위 평균 변화 탐지 : 탐지율 높음
        # df_W_des['week_ratio_mean'] = df_W_des['week_mean'] / df_W_des['week_mean'].shift(1)
        # df_W_des['week_flag_change_mean'] = (df_W_des['week_ratio_mean'] < 0.769) | (df_W_des['week_ratio_mean'] > 1.3)

        # 주단위 표준편차 변화 탐지 : 부하 형태가 달라도 표준편차가 동일한 경우가 있음 -> AND, OR 조건을 잘 따져서 사용해야 함
        # df_W_des['week_ratio_std'] = df_W_des['week_std'] / df_W_des['week_std'].shift(1)
        # df_W_des['week_flag_change_std'] = (df_W_des['week_ratio_std'] < 0.714) | (df_W_des['week_ratio_std'] > 1.4)


        # 주단위 총계 변화 탐지 : 평균에 단순히 7을 곱한 값이기 때문에 평균을 가지고 탐지하는 것과 차이가 없음


        # 주단위 패턴 변화 탐지
        df_W_des = detection_change_week(df_W_des)


        # 주단위 계산 결과 데이터 병합
        df_4H_data = pd.merge(df_4H_data, df_W_des, on='date_group_week', how='left')



        print('경과시간 4:', (time.time() - start))


        df_4H_data['diff_week_max'] = df_4H_data['week_max'].diff()
        df_4H_data['diff_week_mean'] = df_4H_data['week_mean'].diff()
        df_4H_data['diff_week_min'] = df_4H_data['week_min'].diff()


        # 일별 통계량 max
        df_4H_data['yyyymmdd'] = df_4H_data['datetime'].dt.strftime('%Y%m%d')
        df_D_des_max = pd.DataFrame(df_4H_data.groupby('yyyymmdd')['load_max'].describe())
        df_D_des_max['yyyymmdd'] = df_D_des_max.index
        # df_H_max_des_day.rename(columns={'count': 'day_count', 'mean': 'day_mean', 'std': 'day_std', 'min': 'day_min', '25%': 'day_25%', '50%': 'day_50%', '75%': 'day_75%', 'max': 'day_max', 'yyyymmdd': 'yyyymmdd'}, inplace=True)
        df_D_des_max.rename(columns={'max': 'day_max'}, inplace=True)
        df_D_des = pd.DataFrame({'yyyymmdd':df_D_des_max['yyyymmdd'], 'day_max':df_D_des_max['day_max']}, df_D_des_max.index)
        df_D_des['weekday'] = pd.to_datetime(df_D_des['yyyymmdd']).dt.weekday


        # 일별 통계량 mean
        df_D_des_mean = pd.DataFrame(df_4H_data.groupby('yyyymmdd')['load_mean'].describe())
        df_D_des_mean['yyyymmdd'] = df_D_des_mean.index
        df_D_des_mean.rename(columns={'mean': 'day_mean', 'std': 'day_std'}, inplace=True)
        df_D_des['day_mean'] = df_D_des_mean['day_mean']
        df_D_des['day_std'] = df_D_des_mean['day_std']


        # 일별 통계량 min
        df_D_des_min = pd.DataFrame(df_4H_data.groupby('yyyymmdd')['load_min'].describe())
        df_D_des_min['yyyymmdd'] = df_D_des_min.index
        df_D_des_min.rename(columns={'min': 'day_min'}, inplace=True)
        df_D_des['day_min'] = df_D_des_min['day_min']
        df_4H_data = pd.merge(df_4H_data, df_D_des, on='yyyymmdd', how='left')


        # 일단위 범위 계산
        df_D_des['day_range'] = df_D_des['day_max'] - df_D_des['day_min']

        # 일단위 범위 변화 탐지
        df_D_des['day_ratio_range'] = df_D_des['day_range'] / df_D_des['day_range'].shift(7)
        df_D_des['day_flag_change_range'] = (df_D_des['day_ratio_range'] < 0.666) | (df_D_des['day_ratio_range'] > 1.5)

        # 일단위 평균 변화 탐지
        df_D_des['day_ratio_mean'] = df_D_des['day_mean'] / df_D_des['day_mean'].shift(7)
        df_D_des['day_flag_change_mean'] = (df_D_des['day_ratio_mean'] < 0.714) | (df_D_des['day_ratio_mean'] > 1.4)

        # 일단위 표준편차 변화 탐지
        df_D_des['day_ratio_std'] = df_D_des['day_std'] / df_D_des['day_std'].shift(7)
        df_D_des['day_flag_change_std'] = (df_D_des['day_ratio_std'] < 0.666) | (df_D_des['day_ratio_std'] > 1.5)



        df_D_des['diff_day_max'] = df_D_des['day_max'].diff()
        df_D_des['diff_day_mean'] = df_D_des['day_mean'].diff()
        df_D_des['diff_day_min'] = df_D_des['day_min'].diff()
        df_D_des.set_index(pd.to_datetime(df_D_des['yyyymmdd']), inplace=True)

        df_4H_data.set_index(df_4H_data['datetime'], inplace=True)

        print('경과시간 5:', (time.time() - start))


        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
        fig = plt.figure(figsize=(36,14))



    print('경과시간 6:', (time.time() - start))



if __name__ == '__main__':

    dir_source = 'C:\\_data\\부하데이터'

    # 아래 이외 특이 파일 path
    # 수원경기본부_남시화변전소_ID18_남시화_3상 : 중간에 갑자기 부하가 높게 치솟은 후 내려온 데이터
    # 수원경기본부_남시화변전소_ID19_남시화_3상 : 중간 중간 데이터가 갑자기 하락했다 올라오는 데이터
    # 수원경기본부_신덕은변전소_ID2_신덕은_3상 : 중간 중간 데이터가 갑자기 하락했다 올라오는 데이터
    filepaths = ['C:\_data\부하데이터\수원경기본부\남시화변전소\ID20\남시화_3상.xls'
        , 'C:\_data\부하데이터\수원경기본부\금촌변전소\ID15\금촌_3상.xls']

    for filepath in filepaths:
        pqms_change_detection(filepath)




