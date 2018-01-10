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

variable_all = ['TIME_ID', 'POLE_ID', 'SENSOR_ID', 'PART_NAME', 'TEMP', 'HUMI', 'SHOCK', 'PITCH', 'ROLL']
variable_plot = ['TEMP', 'HUMI', 'SHOCK', 'PITCH', 'ROLL']
limit_data = {'AMBIENT':[0, 81900],'BATTERY':[0, 100],'HUMI':[0, 100],'TEMP':[-15, 65],'PITCH':[-180, 180],'ROLL':[-90, 90],'UV':[0, 20.48],'PRESS':[0, 1100],'SHOCK':[0, 1]}
limit_ylim = {'AMBIENT':[-100, 82000], 'BATTERY':[-5, 105], 'HUMI':[-5, 105], 'TEMP':[-15, 65], 'PITCH':[-185, 185], 'ROLL':[-95, 95], 'UV':[-1, 22], 'PRESS':[-10, 1110],'SHOCK':[-1, 2]}


def getPole(df, pole_id, sensor_id, part_name, time_start, time_end, resample_how):

    data_time_start = time_start
    data_time_end = time_end

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
    date_range = pd.date_range(data_time_start, data_time_end, freq=resample_how)
    df = df.reindex(date_range)

    return df


def saveImage(dir_data, dir_output, pole_id, sensor_id, part_name, time_start, time_end, resample_how):
    start_time = time.time()
    df = pd.read_csv('{}{}.csv'.format(dir_data, pole_id))
    df_data = getPole(df, pole_id, sensor_id, part_name, time_start, time_end, resample_how)
    file_name = '{}_{}_{}'.format(pole_id ,sensor_id ,part_name)
    # df_data = df_data.interpolate()
    # print(df_data)
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(file_name)

    index_start = str(df_data.index.values[0])
    index_end = str(df_data.index.values[-1])

    cnt = 0
    for item in variable_plot:
        cnt = cnt + 1
        exec('ax{} = fig.add_subplot({}, 1, {})'.format(str(cnt), str(len(variable_plot)), str(cnt)))
        eval('ax{}.plot(df_data[\'{}\'])'.format(str(cnt), item))
        eval('ax{}.set_xlim([datetime.datetime({}, {}, {}, {}, {}, {}), datetime.datetime({}, {}, {}, {}, {}, {})])'.format(str(cnt), int(index_start[:4]), int(index_start[5:7]), int(index_start[8:10]), int(index_start[11:13]), int(index_start[14:16]), int(index_start[17:19]), int(index_end[:4]), int(index_end[5:7]), int(index_end[8:10]), int(index_end[11:13]), int(index_end[14:16]), int(index_end[17:19])))
        eval('ax{}.set_ylabel(\'{}\')'.format(str(cnt), item))
        eval('ax{}.set_ylim({})'.format(str(cnt), str(limit_ylim[item])))

    print('{}:{}'.format(file_name, str(round(time.time() - start_time, 4))))

    try:
        fig.savefig('{}{}.png'.format(dir_output, file_name), format='png')
        plt.close(fig)
    except Exception as ex:
        traceback.print_exc()
        print('Exception:{}'.format(file_name))
    finally:
        plt.close(fig)

def wrapper_saveImage(args):
    return saveImage(*args)

def main(path_dir, order, resample_how):

    # 데이터 디렉토리 체크
    dir_data = '{}data\\{}\\'.format(path_dir, order)
    if not os.path.isdir(dir_data):
        print('데이터 디렉토리가 존재하지 않습니다.')
        return

    # 아웃풋 디렉토리 체크
    dir_output = '{}output\\'.format(path_dir)
    if not os.path.isdir(dir_output):
        os.mkdir(dir_output)

    df_pole_shock = pd.read_csv('{}iot_pole_1st_shock.csv'.format(path_dir), encoding = "euc-kr")

    cnt = 0
    list_pole = []
    for idx in df_pole_shock.index.values:
        cnt += 1
        print(cnt)
        pole_id = df_pole_shock['POLE_ID'][idx]
        sensor_id = df_pole_shock['SENSOR_ID'][idx]
        part_name = df_pole_shock['PART_NAME'][idx]

        if not os.path.exists('{}{}.csv'.format(dir_data, pole_id)):
            continue

        list_param = []
        list_param.append(dir_data)
        list_param.append(dir_output)
        list_param.append(pole_id)
        list_param.append(sensor_id)
        list_param.append(part_name)
        list_param.append(time_start)
        list_param.append(time_end)
        list_param.append(resample_how)

        list_pole.append(list_param)

    with Pool(processes=5) as pool:
        pool.map(wrapper_saveImage, list_pole)



if __name__ == '__main__':

    start_time_main = time.time()

    order = '1st'
    path_dir = 'F:\\IOT\\'
    time_start = '2016-04-01 00:00:00'
    time_end = '2017-05-15 23:59:59'
    resample_how = '10T'

    main(path_dir, order, resample_how)

    print('Total Time:{}'.format(str(round(time.time() - start_time_main, 4))))

