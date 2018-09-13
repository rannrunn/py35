# coding: utf-8

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import datetime
import time
from multiprocessing import Pool
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


def get_matplotlib_color_list():
    return ['aliceblue','antiquewhite','aqua','aquamarine','azure','beige','bisque','black','blanchedalmond','blue','blueviolet','brown','burlywood','cadetblue','chartreuse','chocolate','coral','cornflowerblue','cornsilk','crimson','cyan','darkblue','darkcyan','darkgoldenrod','darkgray','darkgreen','darkkhaki','darkmagenta','darkolivegreen','darkorange','darkorchid','darkred','darksalmon','darkseagreen','darkslateblue','darkslategray','darkturquoise','darkviolet','deeppink','deepskyblue','dimgray','dodgerblue','firebrick','floralwhite','forestgreen','fuchsia','gainsboro','ghostwhite','gold','goldenrod','gray','green','greenyellow','honeydew','hotpink','indianred','indigo','ivory','khaki','lavender','lavenderblush','lawngreen','lemonchiffon','lightblue','lightcoral','lightcyan','lightgoldenrodyellow','lightgreen','lightgray','lightpink','lightsalmon','lightseagreen','lightskyblue','lightslategray','lightsteelblue','lightyellow','lime','limegreen','linen','magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen','mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','mintcream','mistyrose','moccasin','navajowhite','navy','oldlace','olive','olivedrab','orange','orangered','orchid','palegoldenrod','palegreen','paleturquoise','palevioletred','papayawhip','peachpuff','peru','pink','plum','powderblue','purple','red','rosybrown','royalblue','saddlebrown','salmon','sandybrown','seagreen','seashell','sienna','silver','skyblue','slateblue','slategray','snow','springgreen','steelblue','tan','teal','thistle','tomato','turquoise','violet','wheat','white','whitesmoke','yellow','yellowgreen']


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


def remove_abnormal_data(df):
    # --- delete abnormal data --- #
    """
    1) minus value
    """
    row_idx = 0
    the_num_of_rows = df.shape[row_idx]

    # remove data with minus value
    same_value_idxs = list()
    for idx in range(the_num_of_rows):
        if df['load'][idx] < 0:
            same_value_idxs.append(idx)

    # remove data with consecutive same value
    for idx in range(the_num_of_rows-1):
        if df['load'][idx] == df['load'][idx+1]:
            same_value_idxs.append(idx+1)
    abnormal_ratio = round(len(same_value_idxs)*100/the_num_of_rows, 2)

    print("abnormal_ratio:", abnormal_ratio, "%")

    for idx in same_value_idxs:
        df['load'][idx] = np.nan

    return df


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

def set_dict_group(group_num=np.nan, class_num=np.nan):
    return {'group_num':group_num, 'class_num':class_num}


def get_group_last_week_range(df, dict_group):
    return df.loc[df['group_num'] == dict_group['group_num'], 'week_range'].values[-1]


def get_group_pred_after_week_range(df, class_num, alpha):
    return exponential_smoothing(df.loc[df['class_num'] == class_num, 'week_range'].values, alpha)


def get_group_pred_week_mean_of_maxmin(df, class_num, alpha):
    return exponential_smoothing(df.loc[df['class_num'] == class_num, 'week_mean_of_maxmin'].values, alpha)


def calculate_threshold_fixed_data(threshold_fixed_data, week_mean_of_maxmin, pred_week_range):
    if week_mean_of_maxmin > 8000:
        week_mean_of_maxmin = 8000
    if pred_week_range > 6000:
        pred_week_range = 6000
    threshold_fixed_data = (threshold_fixed_data * (1 - 0.30 * (1 - 0.6 * week_mean_of_maxmin / 8000 - 0.4 * pred_week_range / 6000)))
    return threshold_fixed_data


def calculate_threshold_range_max(threshold_range_max, week_mean_of_maxmin):
    if week_mean_of_maxmin > 8000:
        week_mean_of_maxmin = 8000
    return threshold_range_max - 0.5 * (week_mean_of_maxmin / 8000)


def calculate_threshold_range_min(threshold_range_min, week_mean_of_maxmin):
    if week_mean_of_maxmin > 8000:
        week_mean_of_maxmin = 8000
    return threshold_range_min + 0.4 * (week_mean_of_maxmin / 8000)


def calculate_change_point(df, dict_parameter, list_class_num, week_num_start, week_num, week_range, week_mean_of_maxmin):

    score_change_point = 0

    alpha = dict_parameter['alpha']
    threshold_ratio_range_min = dict_parameter['threshold_ratio_range_min']
    threshold_ratio_range_max = dict_parameter['threshold_ratio_range_max']
    threshold_fixed_week_range = dict_parameter['threshold_fixed_week_range']
    threshold_fixed_week_mean_of_maxmin = dict_parameter['threshold_fixed_week_mean_of_maxmin']


    threshold_ratio_range_min = calculate_threshold_range_min(threshold_ratio_range_min, week_mean_of_maxmin)
    threshold_ratio_range_max = calculate_threshold_range_max(threshold_ratio_range_max, week_mean_of_maxmin)

    sr_bool_index = (df['group_week'] >= week_num_start) & (df['group_week'] < week_num)
    df_temp = df.loc[sr_bool_index]
    df_temp.index = range(0, len(df_temp))

    train_week_range = df_temp['week_range'].values
    train_week_mean_of_maxmin = df_temp['week_mean_of_maxmin'].values

    # 이전 주들의 데이터를 이용해 현재 주의 week_range 예측
    pred_week_range = exponential_smoothing(train_week_range, alpha)
    pred_week_mean_of_maxmin = exponential_smoothing(train_week_mean_of_maxmin, alpha)

    df.loc[(df['group_week'] == week_num), 'pred_week_range'] = pred_week_range

    ratio_load_range = week_range / pred_week_range

    threshold_fixed_week_range = calculate_threshold_fixed_data(threshold_fixed_week_range, week_mean_of_maxmin, pred_week_range)
    threshold_fixed_week_mean_of_maxmin = calculate_threshold_fixed_data(threshold_fixed_week_mean_of_maxmin, week_mean_of_maxmin, pred_week_range)


    # 1. 범위 변화 비율을 이용해 탐지
    # 2. 고정 스케일의 차이를 이용해 탐지
    # 3-1. 현재 class보다 이전 class에 가까운 경우에 탐지 => 시도했으나 매시점을 change point로 탐지하고 전체 데이터를 범위가 같은 주끼리 묶는 결과가 나와 이 함수 내에서는 사용하지 않는 것으로 결정
    check_range_using_ratio = ratio_load_range < threshold_ratio_range_min or ratio_load_range > threshold_ratio_range_max
    check_range_using_fixed_range = np.abs(week_range - pred_week_range) > threshold_fixed_week_range
    check_mean_using_fixed_mean = np.abs(pred_week_mean_of_maxmin - week_mean_of_maxmin) > threshold_fixed_week_mean_of_maxmin
    check_mean_using_range = np.abs(pred_week_mean_of_maxmin - week_mean_of_maxmin) > week_range / 2

    if (check_range_using_ratio and check_range_using_fixed_range) or (check_mean_using_fixed_mean) or (check_mean_using_range):
        score_change_point = 110


    return df, score_change_point


def calculate_class_week(df_W_des, dict_parameter, class_num, week_range, week_mean_of_maxmin):

    score_class = 0

    alpha = dict_parameter['alpha']
    threshold_ratio_range_min = dict_parameter['threshold_ratio_range_min']
    threshold_ratio_range_max = dict_parameter['threshold_ratio_range_max']
    threshold_fixed_week_range = dict_parameter['threshold_fixed_week_range']
    threshold_fixed_week_mean_of_maxmin = dict_parameter['threshold_fixed_week_mean_of_maxmin']
    difference_ratio_mean_of_maxmin = dict_parameter['difference_ratio_mean_of_maxmin']


    threshold_ratio_range_min = calculate_threshold_range_min(threshold_ratio_range_min, week_mean_of_maxmin)
    threshold_ratio_range_max = calculate_threshold_range_max(threshold_ratio_range_max, week_mean_of_maxmin)


    group_pred_after_week_range = get_group_pred_after_week_range(df_W_des, class_num, alpha)
    group_pred_week_mean_of_maxmin = get_group_pred_week_mean_of_maxmin(df_W_des, class_num, alpha)


    threshold_fixed_week_range = calculate_threshold_fixed_data(threshold_fixed_week_range, week_mean_of_maxmin, group_pred_after_week_range)
    threshold_fixed_week_mean_of_maxmin = calculate_threshold_fixed_data(threshold_fixed_week_mean_of_maxmin, week_mean_of_maxmin, group_pred_after_week_range)


    check_range_using_ratio = (week_range / group_pred_after_week_range) >= threshold_ratio_range_min and (week_range / group_pred_after_week_range) <= threshold_ratio_range_max
    check_range_using_fixed_range = np.abs(week_range - group_pred_after_week_range) <= threshold_fixed_week_range
    check_mean_using_fixed_mean = np.abs(group_pred_week_mean_of_maxmin - week_mean_of_maxmin) <= threshold_fixed_week_mean_of_maxmin
    check_mean_using_range = np.abs(group_pred_week_mean_of_maxmin - week_mean_of_maxmin) <= week_range / difference_ratio_mean_of_maxmin


    if (check_range_using_ratio or check_range_using_fixed_range) and (check_mean_using_fixed_mean) and (check_mean_using_range):
        score_class = 110

    sum_of_condition =  np.abs(week_range - group_pred_after_week_range) + np.abs(group_pred_week_mean_of_maxmin - week_mean_of_maxmin) + np.abs(group_pred_week_mean_of_maxmin - week_mean_of_maxmin)

    return score_class, sum_of_condition





def weighted_dict_parameter_week(dict_parameter):
    return_dict_parameter = dict_parameter.copy()
    return_dict_parameter['threshold_ratio_range_min'] = return_dict_parameter['threshold_ratio_range_min'] * 0.8
    return_dict_parameter['threshold_ratio_range_max'] = return_dict_parameter['threshold_ratio_range_max'] * 1.2
    return_dict_parameter['threshold_fixed_week_range'] = return_dict_parameter['threshold_fixed_week_range'] * 1.2
    return_dict_parameter['threshold_fixed_week_mean_of_maxmin'] = return_dict_parameter['threshold_fixed_week_mean_of_maxmin'] * 1.2
    return_dict_parameter['difference_ratio_mean_of_maxmin'] = return_dict_parameter['difference_ratio_mean_of_maxmin'] * 0.8
    return return_dict_parameter



def weighted_dict_parameter_day(dict_parameter):
    return_dict_parameter = dict_parameter.copy()
    return_dict_parameter['threshold_ratio_range_min'] = return_dict_parameter['threshold_ratio_range_min'] * 1.4
    return_dict_parameter['threshold_ratio_range_max'] = return_dict_parameter['threshold_ratio_range_max'] * 0.6
    return_dict_parameter['threshold_fixed_week_range'] = return_dict_parameter['threshold_fixed_week_range'] * 0.6
    return_dict_parameter['threshold_fixed_week_mean_of_maxmin'] = return_dict_parameter['threshold_fixed_week_mean_of_maxmin'] * 0.6
    return_dict_parameter['difference_ratio_mean_of_maxmin'] = return_dict_parameter['difference_ratio_mean_of_maxmin'] * 1.4
    return return_dict_parameter



# 탐지된 클래스 결과를 가지고 재조정 조정이 일어날 경우 조정이 일어나는 지점부터 마지막까지 모두 클래스 변경
def reconstruction_class(df_W_des, dict_parameter):
    list_group_num = list(df_W_des.drop_duplicates(['group_num'])['group_num'].values)
    list_group_num.sort()
    for item_group_num in range(1, len(list_group_num)):
        df_group_now = df_W_des[df_W_des['group_num'] == item_group_num]

        df_group_before = df_W_des.loc[df_W_des['group_num'] < item_group_num]

        list_class_num_group_before = list(df_group_before.drop_duplicates(['class_num'])['class_num'].values)
        list_class_num_group_before.sort()

        for item_class_num in list_class_num_group_before:
            # 아래 Data Frame은 이전 그룹에 대한 클래스 범위라는 것을 잊지 말아라
            df_class_before = df_group_before.loc[df_group_before['class_num'] == item_class_num]
            list_index = list(df_group_now.index)
            total_pass_class_score = 0
            bool_first_pass_class = False
            for idx in range(len(list_index)):
                sr_group_now_by_index = df_group_now.loc[list_index[idx]]
                week_range = sr_group_now_by_index['week_range']
                week_mean_of_maxmin = sr_group_now_by_index['week_mean_of_maxmin']

                # 첫번째
                if idx == 0 :
                    score_class, _ = calculate_class_week(df_class_before, weighted_dict_parameter_week(dict_parameter), item_class_num, week_range, week_mean_of_maxmin)
                    if score_class > 100:
                        bool_first_pass_class = True

                # 첫번째 이후
                if idx > 0:
                    _, class_before_sum_of_condition = calculate_class_week(df_class_before, dict_parameter, item_class_num, week_range, week_mean_of_maxmin)
                    _, class_now_sum_of_condition = calculate_class_week(df_group_now.loc[np.min(df_group_now.index.values):np.min(df_group_now.index.values) + 1], dict_parameter, df_group_now['class_num'].max(), week_range, week_mean_of_maxmin)
                    if class_now_sum_of_condition > class_before_sum_of_condition:
                        total_pass_class_score += 1

            # 현재 로직은 if 조건이 모든 클래스에 적용되지만 결국엔 마지막 클래스만 영향을 미칠 수 밖에 없음
            if total_pass_class_score > 0 and total_pass_class_score == len(df_group_now) - 1:
                df_W_des.loc[np.min(df_group_now.index.values) + 7:np.max(df_group_now.index.values), 'class_num'] = item_class_num
                if bool_first_pass_class == True:
                    df_W_des.loc[np.min(df_group_now.index.values), 'class_num'] = item_class_num


    return df_W_des


# 순방향 탐지 및 패턴 그룹 분류 진행
def detection_change_week(df_W_des, dict_parameter):

    if len(df_W_des) == 0:
        print('패턴 분류를 위한 데이터가 없습니다.')
        return df_W_des

    df_W_des['flag_change_week'] = False

    week_num_start = df_W_des['group_week'].min()
    group_num = 0
    class_num = 0
    df_W_des.loc[(df_W_des['group_week'] == week_num_start), 'group_num'] = group_num
    df_W_des.loc[(df_W_des['group_week'] == week_num_start), 'class_num'] = class_num
    list_group = []
    list_group.append(set_dict_group(group_num, class_num))
    for week_num in range(df_W_des['group_week'].min() + 7, df_W_des['group_week'].max() + 1, 7):

        week_mean_of_maxmin = df_W_des.loc[(df_W_des['group_week'] == week_num), 'week_mean_of_maxmin'].values[0]
        week_range = df_W_des.loc[(df_W_des['group_week'] == week_num), 'week_range'].values[0]

        list_class_num = [item['class_num'] for item in list_group]
        list_class_num.sort()

        df_W_des, score_change_point = calculate_change_point(df_W_des, dict_parameter, list_class_num, week_num_start, week_num, week_range, week_mean_of_maxmin)

        if score_change_point > 100:
            df_W_des.loc[(df_W_des['group_week'] == week_num), 'flag_change_week'] = True
            week_num_start = week_num

            group_num += 1
            class_num = group_num

            # 이전 class 들의 마지막 예측 값과의 차이를 이용해 현재 class를 재구분
            for item_class_num in list_class_num:
                score_class, _ = calculate_class_week(df_W_des, dict_parameter, item_class_num, week_range, week_mean_of_maxmin)
                if score_class > 100:
                    class_num = item_class_num
            # 삽입
            list_group.append(set_dict_group(group_num, class_num))

        df_W_des.loc[(df_W_des['group_week'] == week_num), 'group_num'] = group_num
        df_W_des.loc[(df_W_des['group_week'] == week_num), 'class_num'] = class_num
        # 갱신
        list_group[group_num] = set_dict_group(group_num, class_num)


    print(df_W_des[['week_range', 'pred_week_range']])
    print(list_group)

    df_W_des = reconstruction_class(df_W_des, dict_parameter)

    return df_W_des


def set_position_in_class(df_W_des):
    list_class_num = list(df_W_des.drop_duplicates(['class_num'])['class_num'].values)
    list_class_num.sort()
    # 주가 클래스에서 차지하는 포지션 설정
    # 한주 : 1
    # 한주이상 : (클래스의) 처음 2, 중간 3, 끝 4
    df_W_des['position_in_class'] = 3
    for item_class_num in list_class_num:
        df_W_des_temp = df_W_des.loc[df_W_des['class_num'] == item_class_num]
        # 한주
        if len(df_W_des_temp) == 1:
            df_W_des.loc[np.min(df_W_des.index.values), 'position_in_class'] = 1
        # 한주이상
        elif len(df_W_des_temp) > 1:
            df_W_des.loc[np.min(df_W_des.index.values), 'position_in_class'] = 2
            df_W_des.loc[np.max(df_W_des.index.values), 'position_in_class'] = 4

    return df_W_des


def calculate_class_day(df_W_des, dict_parameter, class_num, day_range):

    score_class = 0

    alpha = dict_parameter['alpha']
    threshold_ratio_range_min = dict_parameter['threshold_ratio_range_min']
    threshold_ratio_range_max = dict_parameter['threshold_ratio_range_max']
    threshold_fixed_week_range = dict_parameter['threshold_fixed_week_range']

    group_pred_after_week_range = get_group_pred_after_week_range(df_W_des, class_num, alpha)

    check_range_using_ratio = (day_range / group_pred_after_week_range) >= threshold_ratio_range_min and (day_range / group_pred_after_week_range) <= threshold_ratio_range_max
    check_range_using_fixed_range = np.abs(day_range - group_pred_after_week_range) <= threshold_fixed_week_range

    print('1:', (day_range / group_pred_after_week_range))
    print('2:', np.abs(day_range - group_pred_after_week_range))

    if (check_range_using_ratio or check_range_using_fixed_range):
        score_class = 110

    sum_of_condition = 0

    return score_class, sum_of_condition


# 순방향 우선, 순방향 적용 시 변화가 없을 경우 역방향 적용
def detection_change_day(df_4H_data, df_W_des, dict_parameter):
    list_group_num = list(df_W_des.drop_duplicates(['group_num'])['group_num'].values)
    list_group_num.sort()
    if len(list_group_num) > 1:

        # for item_group_num in range(1, len(list_group_num)):
        #     df_group_before = df_W_des.loc[df_W_des['group_num'] < item_group_num]
        #     df_group_now = df_W_des[df_W_des['group_num'] == item_group_num]
        #     length_group_now = len(df_group_now)
        #     if length_group_now == 1:
        #         pass
        #     elif length_group_now > 1:
        #         pass

        df_D_data = df_4H_data.drop_duplicates(['yyyymmdd'])[['yyyymmdd', 'group_num', 'class_num', 'group_week', 'day_max', 'day_min']]

        print(df_D_data)

        for item_group_num in range(1, len(list_group_num)):
            print('\n\n\n\nitem_group_num:', item_group_num)
            df_group = df_D_data[df_D_data['group_num'] == item_group_num]
            df_week = df_group[df_group['group_week'] == df_group['group_week'].min()]



            print('len:', len(df_week))
            print(df_week)

            df_group_before = df_W_des.loc[df_W_des['group_num'] == item_group_num - 1]

            class_num_before = df_group_before['class_num'].values[-1]
            print('class_num_before:::::::::::::', class_num_before)

            bool_change_day = False

            for idx in range(len(df_week)):
                sr_day_now = df_week.iloc[idx]

                day_range_now = sr_day_now['day_max'] - sr_day_now['day_min']
                print('day_range_now:', day_range_now)

                score_class, _ = calculate_class_day(df_group_before, weighted_dict_parameter_day(dict_parameter), class_num_before, day_range_now)

                if score_class > 100:
                    # class 변경
                    print('score:', score_class)
                    df_4H_data.loc[df_4H_data['yyyymmdd'] == sr_day_now['yyyymmdd'], 'class_num'] = class_num_before
                    print('이후 데이터 바뀐 클래스:', df_4H_data.loc[df_4H_data['yyyymmdd'] == sr_day_now['yyyymmdd'], 'class_num'].values)
                    bool_change_day = True
                else:
                    break

            print('ddddddddddddddddddd')

            df_group_before_two = df_D_data[df_D_data['group_num'] == item_group_num - 1]
            df_week_before = df_group_before_two[df_group_before_two['group_week'] == df_group_before_two['group_week'].max()]
            df_group_this = df_W_des.loc[df_W_des['group_num'] == item_group_num]
            class_num_now = df_group_this['class_num'].values[-1]

            print(df_week_before)

            if bool_change_day == True:
                continue

            for idx_reversed in reversed(range(len(df_week_before))):

                sr_day_before = df_week_before.iloc[idx_reversed]

                day_range_before = sr_day_before['day_max'] - sr_day_before['day_min']
                print('day_range_before:', day_range_before)

                df_group_this['key_desc'] = df_group_this.index
                df_group_this_reversed = df_group_this.sort_values(by='key_desc', ascending=False).drop('key_desc', axis=1)

                score_class, _ = calculate_class_day(df_group_this_reversed, weighted_dict_parameter_day(dict_parameter), class_num_now, day_range_before)

                if score_class > 100:
                    # class 변경
                    print('score_before:', score_class)
                    df_4H_data.loc[df_4H_data['yyyymmdd'] == sr_day_before['yyyymmdd'], 'class_num'] = class_num_now
                    print('이전 데이터 바뀐 클래스:', df_4H_data.loc[df_4H_data['yyyymmdd'] == sr_day_before['yyyymmdd'], 'class_num'].values)
                else:
                    break


            print('group_break')

    return df_4H_data


    # 두 주씩 윈도우 이동하면서 알고리즘 적용
    # if len(df_W_des) > 1:
    #     for idx in range(1, len(df_W_des)):
    #         sr_temp_now = df_W_des.iloc[idx]
    #         sr_temp_before = df_W_des.iloc[idx - 1]
    #         bool_stop_forward = False
    #         bool_stop_backward = False
    #         # 조건 분기는 많으나 실제 사용하는 분기는 적으니 숙지 필요
    #         if sr_temp_before['postion_in_class'] == 1:
    #             # 현재 가능한 포지션은 1 or 2
    #             if sr_temp_now['postion_in_class'] == 1:
    #                 # 이 부분은 for 문이 한번 끝나고 처리해야 하는 부분이라 패스
    #                 # for문이 끝나면 구현해야 하며 나중에 빈도가 높지 않고 현재 테스트 케이스가 없어 나중에 구현할 예정
    #                 pass
    #             elif sr_temp_now['postion_in_class'] == 2:
    #                 # forward, backward 모두 적용
    #                 # 순방향 우선, 순방향 적용 시 변화가 없을 경우 역방향 적용
    #
    #
    #         elif sr_temp_before['postion_in_class'] == 2:
    #             # 현재 가능한 포지션은 3 or 4
    #             # 3 or 4 모두 현재 포지션이 이전 포지션에 미치는 영향 또는 이전 포지션이 현재 포지션에 미치는 영향이 없으므로 pass
    #             pass
    #         elif sr_temp_before['postion_in_class'] == 3:
    #             # 현재 가능한 포지션은 4
    #             # 현재 포지션이 이전 포지션에 미치는 영향 또는 이전 포지션이 현재 포지션에 미치는 영향이 없으므로 pass
    #             pass
    #         elif sr_temp_before['postion_in_class'] == 4:
    #             # 현재 가능한 포지션은 1 or 2
    #             if sr_temp_now['postion_in_class'] == 1:
    #
    #             elif sr_temp_now['postion_in_class'] == 2:



    return



def pqms_change_detection(args):

    flag_plot = args[0]
    filepath = args[1]
    dir_output = args[2]

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

    # 정영재 연구원이 만든 이상 데이터 탐지 : 시간이 좀 걸려 주석 처리
    # df_temp = remove_abnormal_data(df_temp)


    dict_parameter = {}
    dict_parameter['alpha'] = 0.6
    dict_parameter['threshold_ratio_range_min'] = 0.625
    dict_parameter['threshold_ratio_range_max'] = 1.6
    dict_parameter['threshold_fixed_week_range'] = 1750
    dict_parameter['threshold_fixed_week_mean_of_maxmin'] = 1500
    dict_parameter['difference_ratio_mean_of_maxmin'] = 2


    print('경과시간 1:', (time.time() - start))
    bool_possible = is_possible(df_temp, 'load')
    if bool_possible:

        df_temp.set_index(pd.to_datetime(df_temp[df_temp.columns[0]]), inplace=True)


        df_4H_mean = df_temp.resample('4H').mean()
        df_4H_data = pd.DataFrame()
        df_4H_data['load_mean'] = df_4H_mean['load']


        # 주별 통계량 max
        df_4H_data['datetime'] = df_4H_data.index.values
        df_4H_data['weekday'] = df_4H_data['datetime'].dt.weekday
        df_4H_data['day_of_year'] = df_4H_data['datetime'].dt.dayofyear
        df_4H_data['day_of_years'] = df_4H_data['datetime'].apply(lambda x: df_4H_data.loc[x, 'day_of_year'] + sum([datetime.date(item, 12, 31).timetuple().tm_yday for item in list(set(df_4H_data['datetime'].dt.year.values)) if item < df_4H_data.loc[x, 'datetime'].year]))


        df_4H_data['group_week'] = df_4H_data['datetime'].apply(lambda x: df_4H_data.loc[x, 'day_of_years'] - (df_4H_data.loc[x, 'weekday'] + 1))
        df_4H_data = df_4H_data[(df_4H_data['group_week'] != df_4H_data['group_week'].min()) & (df_4H_data['group_week'] != df_4H_data['group_week'].max())]


        df_4H_data['week_odd'] = df_4H_data.loc[df_4H_data['group_week'] % 2 == 0, 'load_mean']
        df_4H_data['week_even'] = df_4H_data.loc[df_4H_data['group_week'] % 2 == 1, 'load_mean']


        print('경과시간 2:', (time.time() - start))


        # 주별 통계량 mean
        df_W_des = pd.DataFrame(df_4H_data.groupby('group_week')['load_mean'].describe())
        df_W_des.index.name = ''
        df_W_des['group_week'] = df_W_des.index
        df_W_des.rename(columns={'max': 'week_max', 'min': 'week_min', 'mean': 'week_mean', 'std': 'week_std'}, inplace=True)



        print('경과시간 3:', (time.time() - start))


        # 주단위 범위 계산
        df_W_des['week_range'] = df_W_des['week_max'] - df_W_des['week_min']

        # 주단위 최대최소의 평균값 계산
        df_W_des['week_mean_of_maxmin'] = df_W_des['week_max'] - df_W_des['week_range'] / 2


        # 주단위 패턴 변화 탐지
        # 순방향 탐지
        df_W_des = detection_change_week(df_W_des, dict_parameter)
        # df_W_des = set_position_in_class(df_W_des)


        # 주단위 계산 결과 데이터 병합
        df_4H_data = pd.merge(df_4H_data, df_W_des, on='group_week', how='left')


        print('경과시간 4:', (time.time() - start))


        # 일별 통계량 mean
        df_4H_data['yyyymmdd'] = df_4H_data['datetime'].dt.strftime('%Y%m%d')
        df_D_des = pd.DataFrame(df_4H_data.groupby('yyyymmdd')['load_mean'].describe())
        df_D_des.index.name = ''
        df_D_des['yyyymmdd'] = df_D_des.index
        df_D_des.rename(columns={'max': 'day_max', 'min': 'day_min', 'mean': 'day_mean', 'std': 'day_std'}, inplace=True)
        df_D_des['weekday'] = pd.to_datetime(df_D_des['yyyymmdd']).dt.weekday


        # 4시간 단위 데이터와 일단위 데이터 병합
        df_4H_data = pd.merge(df_4H_data, df_D_des, on='yyyymmdd', how='left')
        df_D_des.set_index(pd.to_datetime(df_D_des['yyyymmdd']), inplace=True)
        df_4H_data.set_index(df_4H_data['datetime'], inplace=True)


        # 일단위 클래스 부여
        # df_4H_data = detection_change_day(df_4H_data, df_W_des, dict_parameter)



        print('경과시간 5:', (time.time() - start))


        colors = ['blue', 'orange', 'blue', 'green', 'blue', 'indigo', 'violet', 'gray', 'black', 'pink', 'brown', 'darkgray', 'darkgreen']

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
        ax0_0.plot(df_4H_data['week_odd'], 'b')
        ax0_0.plot(df_4H_data['week_even'], 'r')
        ax0_0.plot(df_4H_data['week_max'])
        ax0_0.plot(df_4H_data['week_mean_of_maxmin'])
        ax0_0.plot(df_4H_data['week_min'])
        ax0_0.set_ylim(ymin=0, ymax=12000)


        ax1_0.plot(df_4H_data.index, df_4H_data['flag_change_week'])
        ax1_0.set_ylim(ymin=0)
        ax1_0.legend()


        ax2_0.plot(df_4H_data.index, df_4H_data['group_num'])
        ax2_0.set_ylim(ymin=-1)
        ax2_0.legend()


        ax0_1.plot(df_4H_data.index, df_4H_data['class_num'])
        ax0_1.set_ylim(ymin=-1)
        ax0_1.legend()


        for idx in range(int(df_W_des['class_num'].max()) + 1):
            df_temp_plot = pd.DataFrame(index=df_4H_data.index)
            df_temp_plot['load_mean'] = df_4H_data.loc[df_4H_data['class_num'] == idx, 'load_mean']
            ax1_1.plot(df_temp_plot['load_mean'], color=colors[idx])
        ax1_1.set_ylim(ymin=-1, ymax=12000)


        for idx in range(int(df_W_des['group_num'].max()) + 1):
            df_temp_plot = pd.DataFrame(index=df_4H_data.index)
            df_temp_plot['load_mean'] = df_4H_data.loc[df_4H_data['group_num'] == idx, 'load_mean']
            ax2_1.plot(df_temp_plot['load_mean'], color=colors[idx])
        ax2_1.set_ylim(ymin=-1, ymax=12000)


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

    flag_plot = 'all'
    dir_source = 'C:\\_data\\부하데이터'
    dir_output = 'C:\\_data\\pqms_change_detection_mmm_week_day_v1\\'
    flag_file = '0' # 0, 1, 2

    filepath = []
    if flag_file == '0':
        filepaths = [
            'C:\_data\부하데이터\수원경기본부\문산변전소\ID7\문산_3상.xls'
        ]

    if flag_file == '1':
        filepaths = [
            'C:\_data\부하데이터\수원경기본부\금촌변전소\ID19\금촌_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\문발변전소\ID4\문발_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\문발변전소\ID8\문발_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\문산변전소\ID17\문산_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\서시화변전소\ID2\서시화_3상.xls'
        ]

    if flag_file == '2':
        filepaths = [
            'C:\_data\부하데이터\수원경기본부\남시화변전소\ID20\남시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\금촌변전소\ID2\금촌_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\금촌변전소\ID3\금촌_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\금촌변전소\ID5\금촌_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\금촌변전소\ID7\금촌_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\금촌변전소\ID9\금촌_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\금촌변전소\ID11\금촌_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\금촌변전소\ID15\금촌_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\금촌변전소\ID16\금촌_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\금촌변전소\ID19\금촌_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\남시화변전소\ID12\남시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\남시화변전소\ID18\남시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\남시화변전소\ID19\남시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\남시화변전소\ID20\남시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\남시화변전소\ID21\남시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\남시화변전소\ID22\남시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\문발변전소\ID3\문발_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\문발변전소\ID4\문발_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\문발변전소\ID6\문발_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\문발변전소\ID8\문발_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\문발변전소\ID10\문발_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\문발변전소\ID12\문발_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\문산변전소\ID7\문산_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\문산변전소\ID8\문산_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\문산변전소\ID10\문산_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\문산변전소\ID14\문산_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\문산변전소\ID16\문산_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\문산변전소\ID17\문산_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\문산변전소\ID18\문산_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\문산변전소\ID19\문산_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\북시화변전소\ID18\북시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\북시화변전소\ID19\북시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\북시화변전소\ID21\북시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\북시화변전소\ID22\북시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\북시화변전소\ID24\북시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\북시화변전소\ID25\북시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\서시화변전소\ID2\서시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\서시화변전소\ID7\서시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\서시화변전소\ID9\서시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\서시화변전소\ID15\서시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\서시화변전소\ID16\서시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\서시화변전소\ID21\서시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\서시화변전소\ID22\서시화_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\신덕은변전소\ID2\신덕은_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\율현변전소\ID2\율현_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\율현변전소\ID6\율현_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\율현변전소\ID9\율현_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\율현변전소\ID10\율현_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\이천변전소\ID4\인천_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\이천변전소\ID9\인천_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\이천변전소\ID13\인천_3상.xls'
            , 'C:\_data\부하데이터\수원경기본부\이천변전소\ID14\인천_3상.xls'
        ]



    for filepath in filepaths:
        pqms_change_detection([flag_plot, filepath, dir_output])
