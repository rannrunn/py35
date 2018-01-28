# coding: utf-8
import os
import time
import traceback

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager, rc
from multiprocessing import Pool

import datetime
from dateutil.relativedelta import relativedelta

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# plot을 위한 변수 리스트
variable_plot = ['TEMP', 'HUMI', 'PITCH', 'ROLL', 'AMBIENT', 'UV', 'PRESS', 'BATTERY', 'SHOCK', 'USN', 'NTC', 'UVC']
variable_all = ['TIME_ID', 'POLE_ID', 'SENSOR_ID', 'PART_NAME'] + variable_plot
limit_data = {'TEMP':[-15, 65], 'HUMI':[0, 100], 'PITCH':[-180, 180], 'ROLL':[-90, 90], 'AMBIENT':[0, 81900], 'UV':[0, 20.48], 'PRESS':[0, 1100], 'BATTERY':[0, 100], 'SHOCK':[0, 1], 'USN':[0, 3000], 'NTC':[-20, 120], 'UVC':[0, 5]}
limit_ylim = {'TEMP':[-15, 15], 'HUMI':[-5, 105], 'PITCH':[-185, 185], 'ROLL':[-95, 95], 'AMBIENT':[-100, 32000], 'UV':[-2, 23], 'PRESS':[-10, 1110], 'BATTERY':[-5, 105], 'SHOCK':[-1, 2], 'USN':[-10, 290], 'NTC':[-25, 125], 'UVC':[-1, 6]}


def process(variable, dir_data, dir_output, period_image, pole_id, time_start, time_end, resample_how, cnt_data):

    df_pole = getTransformerData(dir_data, pole_id, time_start, time_end, resample_how)
    df_pole = removeErrorData(df_pole)
    df_pole = removeOutOfRange(df_pole)

    # 데이터의 기간을 설정
    date_range = pd.date_range(time_start, time_end, freq=resample_how)
    df_pole = df_pole.reindex(date_range)
    step_day = getStepDay(resample_how)

    index_start = datetime.datetime.strptime(str(df_pole.index.values[0])[:19], '%Y-%m-%dT%H:%M:%S')
    index_end = datetime.datetime.strptime(str(df_pole.index.values[-1])[:19], '%Y-%m-%dT%H:%M:%S')
    # 옵션에 데이터 따라 이미지 저장
    # 모든 데이터
    if period_image == 'a':
        # 무조건 이미지 저장
        saveImage(df_pole, dir_output, period_image, pole_id, cnt_data)
    # 월별 데이터
    elif period_image == 'm':
        range_start = datetime.datetime(index_start.year, index_start.month, 1)
        range_end = range_start + relativedelta(months=1) - datetime.timedelta(seconds=1)
        while range_start < index_end:
            df_data = df_pole.loc[range_start:range_end]
            # 데이터가 한개라도 있을 경우 이미지 저장
            if len(df_data.loc[df_data['TEMP'].notnull()]) != 0:
                saveImage(df_data, dir_output, period_image, pole_id, cnt_data)
            range_start += relativedelta(months=1)
            range_end = range_start + relativedelta(months=1) - datetime.timedelta(seconds=1)
    # 주별 데이터
    elif period_image == 'w':
        step_week = step_day * 7
        range_start = datetime.datetime(index_start.year, index_start.month, index_start.day)
        # 일요일
        while range_start.weekday() != 6:
            range_start += datetime.timedelta(1)
        range_end = range_start + datetime.timedelta(7) - datetime.timedelta(seconds=1)
        while range_start < index_end:
            df_data = df_pole[range_start:range_end]
            # 데이터가 모두 있을 경우 이미지 저장
            # if len(df_data[df_data.iloc[:, 1].notnull()]) == step_week:
            saveImage(df_data, dir_output, period_image, pole_id, cnt_data)
            range_start += datetime.timedelta(days=7)
            range_end += datetime.timedelta(days=7)
    # 일별 데이터
    elif period_image == 'd':
        range_start = datetime.datetime(index_start.year, index_start.month, index_start.day)
        range_end = range_start + datetime.timedelta(1) - datetime.timedelta(seconds=1)
        while range_start < index_end:
            df_data = df_pole[range_start:range_end]
            # 데이터가 모두 있을 경우 이미지 저장
            # if len(df_data[df_data.iloc[:, 1].notnull()]) == step_day:
            saveImage(df_data, dir_output, period_image, pole_id, cnt_data)
            range_start += datetime.timedelta(days=1)
            range_end += datetime.timedelta(days=1)

# resample_how에 따른 step을 가져온다.
def getStepDay(resample_how):
    return {
        '10T':144
        ,'60T':24
        ,'120T':12
        ,'1D':1
    }.get(resample_how, 0)

# 전주별 데이터를 가져온다.
def getTransformerData(dir_data, pole_id, time_start, time_end, resample_how):

    df_result = pd.DataFrame(columns=['TIME_ID'])
    df_read = pd.read_csv('{}{}.csv'.format(dir_data, pole_id), encoding='euc-kr')
    df_read = df_read[['TIME_ID', 'POLE_ID', 'SENSOR_ID', 'PART_NAME', 'TEMP']]
    df_sensor = df_read[['SENSOR_ID', 'PART_NAME']].drop_duplicates()

    pole_sensor = ''
    df_pole_pole = pd.DataFrame()
    for idx in df_sensor.index.values:
        sensor_id = str(df_sensor['SENSOR_ID'][idx])
        part_name = str(df_sensor['PART_NAME'][idx])
        if part_name == '전주':
            sensor_id_pole = sensor_id
            df_pole_pole = df_read[df_read['SENSOR_ID'] == sensor_id_pole][['TIME_ID', 'TEMP']]
            df_pole_pole['TEMP'] = df_pole_pole['TEMP'].astype(float)
            df_pole_pole.set_index('TIME_ID', inplace=True)
            df_pole_pole.index = pd.to_datetime(df_pole_pole.index)
            df_pole_pole = df_pole_pole.resample(resample_how).mean()
            print(df_pole_pole)

    for idx in df_sensor.index.values:
        sensor_id = str(df_sensor['SENSOR_ID'][idx])
        part_name = str(df_sensor['PART_NAME'][idx])
        if part_name == 'nan' or part_name != '변압기 본체':
            continue
        df_temp = df_read[df_read['SENSOR_ID'] == sensor_id][['TIME_ID', 'TEMP']]
        df_temp['TEMP'] = df_temp['TEMP'].astype(float)
        df_temp.set_index('TIME_ID', inplace=True)
        df_temp.index = pd.to_datetime(df_temp.index)
        df_temp = df_temp.resample(resample_how).mean()
        df_temp = df_temp.subtract(df_pole_pole, axis='index', level=None, fill_value=None)

        df_temp.columns = [sensor_id + '_' + part_name]
        df_result = pd.merge(df_result, df_temp, left_index=True, right_index=True, how='outer')
    # inplace True => 제자리에서 변경(새로운 DataFrame을 생성하지 않음)

    return df_result

# 범위 밖 데이터 제거
def removeOutOfRange(df):
    # 아웃라이어 제거
    for col in df.columns:
        if col in limit_data.keys():
            mask = df[col] < limit_data[col][0]
            df.loc[mask, col] = None

            mask = df[col] > limit_data[col][1]
            df.loc[mask, col] = None
    return df


# 오류 데이터 제거
def removeErrorData(df):
    if 'AMBIENT' in df.columns:
        mask = df['AMBIENT'] == 60000
        df.loc[mask, 'AMBIENT'] = None
    if 'PRESS' in df.columns:
        mask = df['PRESS'] == 100
        df.loc[mask, 'PRESS'] = None
    return df


def saveImage(df, dir_output, period_image, pole_id, cnt_data):

    index_start = datetime.datetime.strptime(str(df.index.values[0])[:19], '%Y-%m-%dT%H:%M:%S')
    index_end = datetime.datetime.strptime(str(df.index.values[-1])[:19], '%Y-%m-%dT%H:%M:%S')

    if period_image == 'a':
        file_name = '{}'.format(pole_id )
    else:
        file_name = '{}_{}'.format(pole_id , str(df.index.values[0])[:10])


    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(file_name)
    # print(df[df['TEMP'].notnull()])
    # df = df.interpolate()

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(df)
    ax1.set_xlim([datetime.datetime(index_start.year, index_start.month, index_start.day, index_start.hour, index_start.minute, index_start.second), datetime.datetime(index_end.year, index_end.month, index_end.day, index_end.hour, index_end.minute, index_end.second)])
    ax1.set_ylim(limit_ylim['TEMP'])
    plt.grid(True)


    print('CNT:{}, {}'.format(cnt_data, file_name))

    try:
        fig.savefig('{}{}.png'.format(dir_output, file_name), format='png')
        plt.close(fig)
    except Exception as ex:
        traceback.print_exc()
        print('Exception:{}'.format(file_name))
    finally:
        plt.close(fig)


def wrapper_process(args):
    return process(*args)

def main(sel_pole_id, pole_top, variable, path_dir, period, period_image, time_start, time_end, resample_how):

    # 데이터 디렉토리 체크
    dir_data = '{}data\\{}\\'.format(path_dir, period)
    if not os.path.isdir(dir_data):
        print('데이터 디렉토리가 존재하지 않습니다.')
        return

    # 아웃풋 디렉토리 체크
    dir_output = '{}output\\{}\\{}_{}\\'.format(path_dir, period, variable, resample_how)
    if not os.path.isdir('{}output\\1\\'.format(path_dir, period)):
        os.mkdir('{}output\\1\\'.format(path_dir, period))
    if not os.path.isdir('{}output\\2\\'.format(path_dir, period)):
        os.mkdir('{}output\\2\\'.format(path_dir, period))
    if not os.path.isdir(dir_output):
        os.mkdir(dir_output)

    df_pole_list = pd.read_csv('{}iot_pole_0_info.csv'.format(path_dir), encoding = "euc-kr")
    df_pole_top = pd.read_csv('{}iot_pole_0_cnt.csv'.format(path_dir), encoding = "euc-kr")

    if period == '1':
        df_pole_list = df_pole_list[df_pole_list['PERIOD'] == 'FIRST']
        if pole_top != 0:
            df_pole_top = df_pole_top[df_pole_top['PERIOD'] == 'FIRST']
            df_pole_top = df_pole_top[:pole_top]
            df_pole_list = df_pole_list[df_pole_list['POLE_ID'].isin(df_pole_top['POLE_ID'])]

    if period == '2':
        df_pole_list = df_pole_list[df_pole_list['PERIOD'] == 'SECOND']
        if pole_top != 0:
            df_pole_top = df_pole_top[df_pole_top['PERIOD'] == 'SECOND']
            df_pole_top = df_pole_top[:pole_top]
            df_pole_list = df_pole_list[df_pole_list['POLE_ID'].isin(df_pole_top['POLE_ID'])]

    # 변압기 센서를 가진 전주
    sr_pole_ts = df_pole_list[df_pole_list['PART_NAME'] == '변압기 본체']['POLE_ID'].unique()
    # 전주 센서를 가진 전주
    sr_pole_ps = df_pole_list[df_pole_list['PART_NAME'] == '전주']['POLE_ID'].unique()
    # 변압기와 전주 센서를 가진 전주의 리스트
    df_pole_list = df_pole_list[df_pole_list['POLE_ID'].isin(sr_pole_ts)]
    df_pole_list = df_pole_list[df_pole_list['POLE_ID'].isin(sr_pole_ps)]

    if sel_pole_id != '':
        df_pole_list = df_pole_list[df_pole_list['POLE_ID'] == sel_pole_id]

    cnt = 0
    list_pole = []
    for idx in df_pole_list.index.values:
        cnt += 1

        pole_id = df_pole_list['POLE_ID'][idx]

        if not os.path.exists('{}{}.csv'.format(dir_data, pole_id)):
            continue

        # dir_data, dir_output, period_image, pole_id, sensor_id, part_name, time_start, time_end, resample_how

        list_param = []
        list_param.append(variable)
        list_param.append(dir_data)
        list_param.append(dir_output)
        list_param.append(period_image)
        list_param.append(pole_id)
        list_param.append(time_start)
        list_param.append(time_end)
        list_param.append(resample_how)
        list_param.append(cnt)

        list_pole.append(list_param)

    print('List Length:{}'.format(cnt))

    with Pool(processes=1) as pool:
        pool.map(wrapper_process, list_pole)



if __name__ == '__main__':

    start_time_main = time.time()

    # 'TEMP', 'HUMI', 'PITCH', 'ROLL', 'AMBIENT', 'UV', 'PRESS', 'BATTERY', 'SHOCK', 'USN', 'NTC', 'UVC'
    pole_top = 50 # plot하려는 전주의 개수(최대는 50까지, 0일 경우에는 모두)
    variable = 'TEMP'
    period = '2'
    period_image = 'w' # all, month, week, day 의 이니셜
    path_dir = 'F:\\IOT\\'
    if period == '1':
        time_start = '2016-04-01 00:00:00'
        time_end = '2017-05-31 23:59:59'
    elif period == '2':
        time_start = '2017-06-01 00:00:00'
        time_end = '2017-12-31 23:59:59'

    resample_how = '10T'

    sel_pole_id = '8232P471'

    print(variable)

    main(sel_pole_id, pole_top, variable, path_dir, period, period_image, time_start, time_end, resample_how)

    print('Total Time:{}'.format(str(round(time.time() - start_time_main, 4))))

