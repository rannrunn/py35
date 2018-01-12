# coding: utf-8
import os
import time
import traceback

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager, rc
from multiprocessing import Pool

import datetime

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

variable_all = ['TIME_ID', 'POLE_ID', 'SENSOR_ID', 'PART_NAME', 'TEMP', 'HUMI', 'PITCH', 'ROLL','AMBIENT','PRESS','USN', 'NTC', 'UVC', 'UV']
variable_plot = ['TEMP', 'HUMI', 'PITCH', 'ROLL','AMBIENT','PRESS','USN', 'NTC', 'UVC', 'UV']
limit_data = {'AMBIENT':[0, 81900],'BATTERY':[0, 100],'HUMI':[0, 100],'TEMP':[-15, 65],'PITCH':[-180, 180],'ROLL':[-90, 90],'UV':[0, 20.48],'PRESS':[0, 1100],'SHOCK':[0, 1],'NTC':[-20, 120],'UVC':[0, 5],'USN':[0, 3000]}
limit_ylim = {'AMBIENT':[-100, 82000], 'BATTERY':[-5, 105], 'HUMI':[-5, 105], 'TEMP':[-15, 65], 'PITCH':[-185, 185], 'ROLL':[-95, 95], 'UV':[-1, 22], 'PRESS':[-10, 1110],'SHOCK':[-1, 2],'NTC':[-22, 122],'UVC':[-1, 6],'USN':[-10, 120]}


def process(variable, dir_data, dir_output, period_image, pole_id, sensor_id, part_name, time_start, time_end, resample_how):

    df_pole = getPole(dir_data, pole_id, sensor_id, part_name, time_start, time_end, resample_how)
    if variable != '':
        df_variable = df_pole[df_pole[variable].notnull()]

        if variable == 'USN':
            df_variable = df_variable[df_variable[variable] != 0]
            df_variable = df_variable[df_variable[variable] != 1]

        if variable == 'AMBIENT':
            df_variable = df_variable[df_variable[variable] > 10]

        if variable == 'AMBIENT':
            df_variable = df_variable[df_variable[variable] > 0]

        if len(df_variable) == 0:
            return

    if period_image == 'a':
        saveImage(df_pole, dir_output, pole_id, sensor_id, part_name)
    elif period_image == 'm':
        pass
    elif period_image == 'd':
        step = 144 # 10T일 경우 144
        for idx in range(len(df_pole), 0, step):
            df_data = df_pole[idx:idx + step]
            if len(df_data.loc[df_data['TEMP'].notnull()]) == step:
                saveImage(df_data, dir_output, pole_id, sensor_id, part_name)


def getPole(dir_data, pole_id, sensor_id, part_name, time_start, time_end, resample_how):

    df = pd.read_csv('{}{}.csv'.format(dir_data, pole_id), encoding='euc-kr')
    df = df[variable_all]
    df = df.loc[df['SENSOR_ID'] == sensor_id]
    df = df.loc[df['PART_NAME'] == part_name]
    # inplace True => 제자리에서 변경(새로운 DataFrame을 생성하지 않음)
    df.set_index('TIME_ID', inplace=True)
    df.index = pd.to_datetime(df.index)

    # 아웃라이어 제거
    for item in variable_plot:
        if item in limit_data.keys():
            mask = df[item] < limit_data[item][0]
            df.loc[mask, item] = None

            mask = df[item] > limit_data[item][1]
            df.loc[mask, item] = None

    # resample을 하기 위해서는 인덱스의 형식이 datetime 형식이어야 한다.
    df = df.resample(resample_how).mean()
    date_range = pd.date_range(time_start, time_end, freq=resample_how)
    df = df.reindex(date_range)

    return df

def saveImage(df, dir_output, pole_id, sensor_id, part_name):

    file_name = '{}_{}_{}'.format(pole_id ,sensor_id ,part_name)
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(file_name)
    # print(df[df['TEMP'].notnull()])
    # df = df.interpolate()
    index_start = str(df.index.values[0])
    index_end = str(df.index.values[-1])

    cnt = 0
    for item in variable_plot:
        cnt = cnt + 1
        exec('ax{} = fig.add_subplot({}, 1, {})'.format(str(cnt), str(len(variable_plot)), str(cnt)))
        eval('ax{}.plot(df[\'{}\'])'.format(str(cnt), item))
        eval('ax{}.set_xlim([datetime.datetime({}, {}, {}, {}, {}, {}), datetime.datetime({}, {}, {}, {}, {}, {})])'.format(str(cnt), int(index_start[:4]), int(index_start[5:7]), int(index_start[8:10]), int(index_start[11:13]), int(index_start[14:16]), int(index_start[17:19]), int(index_end[:4]), int(index_end[5:7]), int(index_end[8:10]), int(index_end[11:13]), int(index_end[14:16]), int(index_end[17:19])))
        eval('ax{}.set_ylabel(\'{}\')'.format(str(cnt), item))
        eval('ax{}.set_ylim({})'.format(str(cnt), str(limit_ylim[item])))

    print('{}'.format(file_name))

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

def main(variable, path_dir, period, period_image, time_start, time_end, resample_how):

    # 데이터 디렉토리 체크
    dir_data = '{}data\\{}\\'.format(path_dir, period)
    if not os.path.isdir(dir_data):
        print('데이터 디렉토리가 존재하지 않습니다.')
        return

    # 아웃풋 디렉토리 체크
    dir_output = '{}output_plot\\'.format(path_dir)
    if not os.path.isdir(dir_output):
        os.mkdir(dir_output)

    df_param_cnt = pd.read_csv('{}iot_pole_000_param_cnt.csv'.format(path_dir), encoding = "euc-kr")

    cnt = 0
    list_pole = []
    for idx in df_param_cnt.index.values:
        cnt += 1

        pole_id = df_param_cnt['POLE_ID'][idx]
        sensor_id = df_param_cnt['SENSOR_ID'][idx]
        part_name = df_param_cnt['PART_NAME'][idx]

        if not os.path.exists('{}{}.csv'.format(dir_data, pole_id)):
            continue

        # dir_data, dir_output, period_image, pole_id, sensor_id, part_name, time_start, time_end, resample_how

        list_param = []
        list_param.append(variable)
        list_param.append(dir_data)
        list_param.append(dir_output)
        list_param.append(period_image)
        list_param.append(pole_id)
        list_param.append(sensor_id)
        list_param.append(part_name)
        list_param.append(time_start)
        list_param.append(time_end)
        list_param.append(resample_how)

        list_pole.append(list_param)

    with Pool(processes=5) as pool:
        pool.map(wrapper_process, list_pole)



if __name__ == '__main__':

    start_time_main = time.time()

    variable = 'UV'
    period = 2
    period_image = 'a' # all, month, day 의 이니셜
    path_dir = 'F:\\IOT\\'
    if period == 1:
        time_start = '2016-04-20 00:00:00'
        time_end = '2017-05-12 23:59:59'
    elif period == 2:
        time_start = '2017-06-17 00:00:00'
        time_end = '2017-12-31 23:59:59'

    resample_how = '120T'

    main(variable, path_dir, period, period_image, time_start, time_end, resample_how)

    print('Total Time:{}'.format(str(round(time.time() - start_time_main, 4))))

