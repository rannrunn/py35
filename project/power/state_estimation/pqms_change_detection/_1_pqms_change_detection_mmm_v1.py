# coding: utf-8

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import datetime
import time
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from multiprocessing import Process

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


# Simple Exponential Smoothing
def exponential_smoothing(series, alpha):

    if len(series) == 0:
        return 0

    list = []
    for i, real in enumerate(reversed(series)):
        weight = alpha * (1 - alpha) ** i
        list.append(real * weight)
    list.append(real * ((1 - alpha) ** (i + 1)))

    return sum(list)


def regularization(df, col, col_reg):
    df[col_reg] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

def set_dict_group(group_num=np.nan, class_num=np.nan, last_week_range=np.nan, last_pred_week_range=np.nan, last_week_mean=np.nan):
    return {'group_num':group_num, 'class_num':class_num, 'last_week_range':last_week_range, 'last_week_mean':last_week_mean}


def get_group_last_week_range(df, dict_group):
    return df.loc[df['group_num'] == dict_group['group_num'], 'week_range'].values[-1]


def get_group_pred_after_week_range(df, dict_group, alpha):
    return exponential_smoothing(df.loc[df['group_num'] == dict_group['group_num'], 'week_range'].values, alpha)


def get_group_last_week_mean_of_minmax(df, dict_group):
    return df.loc[df['group_num'] == dict_group['group_num'], 'week_mean_of_minmax'].values[-1]


def calculate_change_point_rating(df, dict_parameter, week_num_start, week_num, week_range, week_mean_of_minmax):

    score_change_point = 0

    alpha = dict_parameter['alpha']
    threshold_ratio_range_min = dict_parameter['threshold_ratio_range_min']
    threshold_ratio_range_max = dict_parameter['threshold_ratio_range_max']
    threshold_ratio_before_mean = dict_parameter['threshold_ratio_before_mean']
    threshold_ratio_before_mean_two = dict_parameter['threshold_ratio_before_mean_two']
    threshold_fixed_load = dict_parameter['threshold_fixed_load']
    threshold_fixed_mean = dict_parameter['threshold_fixed_mean']

    threshold_ratio_before_mean = cal_threshold_ratio_before_mean(threshold_ratio_before_mean, week_mean_of_minmax)

    sr_bool_index = (df['date_group_week'] >= week_num_start) & (df['date_group_week'] < week_num)

    print('11')

    df_temp = pd.DataFrame()
    df_temp = df.loc[sr_bool_index]
    df_temp.index = range(0, len(df_temp))

    train_week_range = df_temp['week_range'].values
    train_week_max = df_temp['week_max'].values
    train_week_min = df_temp['week_min'].values
    train_week_mean = df_temp['week_mean'].values

    print(train_week_range)


    before_week_mean_of_minmax = df.loc[(df['date_group_week'] == week_num - 7), 'week_mean_of_minmax'].values[0]
    # 이전 주들의 데이터를 이용해 현재 주의 week_range 예측
    pred_week_range = exponential_smoothing(train_week_range, alpha)
    pred_week_max = exponential_smoothing(train_week_range, alpha)
    pred_week_min = exponential_smoothing(train_week_max, alpha)
    pred_week_mean = exponential_smoothing(train_week_min, alpha)

    df.loc[(df['date_group_week'] == week_num), 'pred'] = pred_week_range

    load_range_ratio = week_range / pred_week_range

    # 1. 범위 변화 비율을 이용해 판단
    # 2. 고정 스케일의 차이를 이용해 판단 (추후 적용)
    check_one = load_range_ratio < threshold_ratio_range_min or load_range_ratio > threshold_ratio_range_max
    check_two = np.abs(week_range - pred_week_range) > before_week_mean_of_minmax * threshold_ratio_before_mean
    check_two_five = np.abs(week_range - pred_week_range) > threshold_fixed_load
    # check_one_two = load_range_ratio < threshold_ratio_range_min * 1.2 or load_range_ratio > threshold_ratio_range_max * 0.833333
    check_three = np.abs(before_week_mean_of_minmax - week_mean_of_minmax) > before_week_mean_of_minmax * threshold_ratio_before_mean_two
    check_three_five = np.abs(before_week_mean_of_minmax - week_mean_of_minmax) > threshold_fixed_mean


    # 디텍션 조건 : 최댓값 변화율
    check_four = df.loc[(df['date_group_week'] == week_num), 'pred'] = pred_week_range
    df.loc[(df['date_group_week'] == week_num), 'check_one'] = check_one
    df.loc[(df['date_group_week'] == week_num), 'check_two'] = check_two
    df.loc[(df['date_group_week'] == week_num), 'check_three'] = check_three

    if (check_one and check_two and check_two_five) or (check_three and check_three_five):
        score_change_point = 110

    return df, score_change_point


def calculate_class_rating(df, dict_parameter, list_group, idx, week_range, week_mean_of_minmax):

    score_class = 0

    alpha = dict_parameter['alpha']
    threshold_ratio_range_min = dict_parameter['threshold_ratio_range_min']
    threshold_ratio_range_max = dict_parameter['threshold_ratio_range_max']
    threshold_ratio_before_mean = dict_parameter['threshold_ratio_before_mean']
    threshold_ratio_before_mean_two = dict_parameter['threshold_ratio_before_mean_two']
    threshold_fixed_load = dict_parameter['threshold_fixed_load']
    threshold_fixed_mean = dict_parameter['threshold_fixed_mean']

    threshold_ratio_before_mean = cal_threshold_ratio_before_mean(threshold_ratio_before_mean, week_mean_of_minmax)

    group_pred_after_week_range = get_group_pred_after_week_range(df, list_group[idx], alpha)
    group_last_week_mean_of_minmax = get_group_last_week_mean_of_minmax(df, list_group[idx])
    check_2_one = (week_range / group_pred_after_week_range) >= threshold_ratio_range_min and (week_range / group_pred_after_week_range) <= threshold_ratio_range_max
    check_2_two = np.abs(week_range - group_pred_after_week_range) <= group_last_week_mean_of_minmax * threshold_ratio_before_mean
    check_2_two_five = np.abs(week_range - group_pred_after_week_range) <= threshold_fixed_load
    # check_2_one_two = (week_range / group_pred_after_week_range) >= threshold_ratio_range_min * 0.769 and (week_range / group_pred_after_week_range) <= threshold_ratio_range_max * 1.3
    check_2_three = np.abs(group_last_week_mean_of_minmax - week_mean_of_minmax) <= group_last_week_mean_of_minmax * threshold_ratio_before_mean_two
    check_2_three_five = np.abs(group_last_week_mean_of_minmax - week_mean_of_minmax) <= threshold_fixed_mean

    if (check_2_one or check_2_two or check_2_two_five) and (check_2_three or check_2_three_five):
        score_class = 110

    return score_class


def cal_threshold_ratio_before_mean(threshold_ratio_before_mean, week_mean_of_minmax):
    if week_mean_of_minmax > 10000:
        week_mean_of_minmax = 10000
    return threshold_ratio_before_mean - 0.1 * (week_mean_of_minmax / 10000)




# 순방향 탐지 및 패턴 그룹 분류 진행
def detection_change_week_forward_direction(df):

    if len(df) == 0:
        print('패턴 분류를 위한 데이터가 없습니다.')
        return df

    df['week_flag_change_range_forward'] = False

    dict_parameter = {}
    dict_parameter['alpha'] = 0.7
    dict_parameter['threshold_ratio_range_min'] = 0.6
    dict_parameter['threshold_ratio_range_max'] = 1.5
    dict_parameter['threshold_ratio_before_mean'] = 0.30
    dict_parameter['threshold_ratio_before_mean_two'] = 0.30
    dict_parameter['threshold_fixed_load'] = 1500
    dict_parameter['threshold_fixed_mean'] = 1250



    week_num_start = df['date_group_week'].min()
    group_num = 0
    class_num = 0
    df.loc[(df['date_group_week'] == week_num_start), 'group_num'] = group_num
    df.loc[(df['date_group_week'] == week_num_start), 'class_num'] = class_num
    list_group = []
    list_group.append(set_dict_group(group_num, class_num, df.loc[(df['date_group_week'] == week_num_start), 'week_range'].values[0], df.loc[(df['date_group_week'] == week_num_start), 'week_mean_of_minmax'].values[0]))
    for week_num in range(df['date_group_week'].min() + 7, df['date_group_week'].max() + 1, 7):

        # print('week_num:', week_num)
        # print('range:', len(df.loc[(df['date_group_week'] == week_num), 'week_range']))

        week_mean_of_minmax = df.loc[(df['date_group_week'] == week_num), 'week_mean_of_minmax'].values[0]
        week_range = df.loc[(df['date_group_week'] == week_num), 'week_range'].values[0]

        df, score_change_point = calculate_change_point_rating(df, dict_parameter, week_num_start, week_num, week_range, week_mean_of_minmax)

        if score_change_point > 100:
            df.loc[(df['date_group_week'] == week_num), 'week_flag_change_range_forward'] = True
            week_num_start = week_num

            group_num += 1
            class_num = group_num

            # 이전 class 들의 마지막 예측 값과의 차이를 이용해 현재 class를 재구분
            for idx in range(group_num - 1):

                scoer_class = calculate_class_rating(df, dict_parameter, list_group, idx, week_range, week_mean_of_minmax)
                if scoer_class > 100:
                    class_num = list_group[idx]['class_num']
            # 삽입
            list_group.append(set_dict_group(group_num, class_num, week_range, week_mean_of_minmax))

        df.loc[(df['date_group_week'] == week_num), 'group_num'] = group_num
        df.loc[(df['date_group_week'] == week_num), 'class_num'] = class_num
        # 갱신
        list_group[group_num] = set_dict_group(group_num, class_num, week_range, week_mean_of_minmax)


    print(df[['week_range', 'pred', 'check_one', 'check_two', 'check_three']])
    print(list_group)

    return df



def pqms_change_detection(args):

    flag_plot = args[0]
    filepath = args[1]
    dir_output = args[2]
    flag_one_file = args[3]

    if filepath != 'C:\_data\부하데이터\수원경기본부\문발변전소\ID3\문발_3상.xls' and flag_one_file:
        return

    if not os.path.isdir(dir_output):
        os.makedirs(dir_output)

    start = time.time()

    filename = os.path.basename(filepath)
    dirname = os.path.dirname(filepath)
    output_filename = filepath.replace('C:\\_data\\부하데이터\\', '').replace('\\', '_').replace('.xls', '.png')


    if not filename.find('3상') > -1: #
        return


    df_temp = pd.read_excel(filepath)
    df_temp = df_temp.rename(columns={'Unnamed: 1': 'load'})


    # 값이 0과 같거나 작을 경우 np.nan으로 대체
    # df_temp.loc[df_temp['load'] <= 0, 'load'] = np.nan


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


        df_W_des_max = pd.DataFrame(df_4H_data.groupby('date_group_week')['load_mean'].describe())
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
        df_W_des_min = pd.DataFrame(df_4H_data.groupby('date_group_week')['load_mean'].describe())
        df_W_des_min['date_group_week'] = df_W_des_min.index
        df_W_des_min.rename(columns={'min': 'week_min'}, inplace=True)
        df_W_des['week_min'] = df_W_des_min['week_min']

        # 주단위 범위 계산
        df_W_des['week_range'] = df_W_des['week_max'] - df_W_des['week_min']

        # 주단위 최대최소의 평균값 계산
        df_W_des['week_mean_of_minmax'] = df_W_des['week_max'] - df_W_des['week_range'] / 2

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
        # 순방향 탐지
        df_W_des = detection_change_week_forward_direction(df_W_des)
        # 순방향과 역방향 조합
        df_W_des['week_flag_change_range'] = df_W_des['week_flag_change_range_forward']




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

        df_D_des['diff_day_max'] = df_D_des['day_max'].diff()
        df_D_des['diff_day_mean'] = df_D_des['day_mean'].diff()
        df_D_des['diff_day_min'] = df_D_des['day_min'].diff()
        df_D_des.set_index(pd.to_datetime(df_D_des['yyyymmdd']), inplace=True)

        df_4H_data.set_index(df_4H_data['datetime'], inplace=True)

        print('경과시간 5:', (time.time() - start))



        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'gray', 'black', 'pink', 'brown']

        fig = plt.figure(figsize=(14,9))

        gs = gridspec.GridSpec(3, 2)
        ax0_0 = plt.subplot(gs[0, 0])
        ax1_0 = plt.subplot(gs[1, 0], sharex=ax0_0)
        ax2_0 = plt.subplot(gs[2, 0], sharex=ax1_0)


        ax0_1 = plt.subplot(gs[0, 1], sharex=ax1_0)
        ax1_1 = plt.subplot(gs[1, 1], sharex=ax1_0)
        ax2_1 = plt.subplot(gs[2, 1], sharex=ax1_0)


        # ax2_1.plot(df_data_decom['weekday'] * 100, label='weekday')
        ax0_0.set_title(output_filename)
        ax0_0.plot(df_4H_data['data_week_odd'], 'b')
        ax0_0.plot(df_4H_data['data_week_even'], 'r')
        ax0_0.plot(df_4H_data['week_max'])
        ax0_0.plot(df_4H_data['week_mean_of_minmax'])
        ax0_0.plot(df_4H_data['week_min'])
        ax0_0.set_ylim(ymin=0)


        ax1_0.plot(df_4H_data.index, df_4H_data['week_flag_change_range'])
        ax1_0.set_ylim(ymin=0)
        ax1_0.legend()


        ax2_0.plot(df_4H_data.index, df_4H_data['group_num'])
        ax2_0.set_ylim(ymin=-1)
        ax2_0.legend()


        ax0_1.plot(df_4H_data.index, df_4H_data['class_num'])
        ax0_1.set_ylim(ymin=-1)
        ax0_1.legend()


        for idx in range(int(df_W_des['class_num'].max()) + 1):
            df_temp = pd.DataFrame(index=df_4H_data.index)
            df_temp['load_mean'] = df_4H_data.loc[df_4H_data['class_num'] == idx, 'load_mean']
            ax1_1.plot(df_temp['load_mean'], color=colors[idx])
        ax1_1.set_ylim(ymin=-1)



        for ax in fig.axes:
            plt.sca(ax)
            plt.xticks(rotation=30)

        # xticks를 잘리지 않고 출력하기 위한 코드 : plt.tight_layout()
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        if flag_plot == 'save':
            plt.savefig(dir_output + output_filename, bbox_inches='tight')
        elif flag_plot == 'show':
            plt.show()
        elif flag_plot == 'all':
            plt.savefig(dir_output + output_filename, bbox_inches='tight')
            plt.show()
        plt.close()

    print('경과시간 6:', (time.time() - start))

if __name__ == '__main__':

    flag_plot = 'save' # save, show
    dir_source = 'C:\\_data\\부하데이터'
    dir_output = 'C:\\_data\\pqms_change_detection_mmm_all_v1\\'
    flag_one_file = True

    names = []
    for path, dirs, filenames in os.walk(dir_source):
        for filename in filenames:
            names.append([flag_plot, os.path.join(path, filename), dir_output, flag_one_file])

    print(names)

    with Pool(processes=14) as pool:
        pool.map(pqms_change_detection, names)





