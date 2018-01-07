# coding: utf-8
import os
import time
import traceback

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager, rc
from multiprocessing import Pool


font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

variable_all = ['TIME_ID', 'POLE_ID', 'SENSOR_ID', 'PART_NAME', 'AMBIENT', 'BATTERY', 'HUMI', 'TEMP', 'PITCH', 'ROLL', 'UV', 'PRESS']
variable_plot = ['AMBIENT', 'BATTERY', 'HUMI', 'TEMP', 'PITCH', 'ROLL', 'UV', 'PRESS']
limit_data = {'AMBIENT':[0, 81900],'BATTERY':[0, 100],'HUMI':[0, 100],'TEMP':[-15, 65],'PITCH':[-180, 180],'ROLL':[-90, 90],'UV':[0, 20.48],'PRESS':[0, 1100]}
limit_ylim = {'AMBIENT':[-100, 82000], 'BATTERY':[-5, 105], 'HUMI':[-5, 105], 'TEMP':[-15, 65], 'PITCH':[-185, 185], 'ROLL':[-95, 95], 'UV':[-1, 22], 'PRESS':[-10, 1110]}


def getPole(df, pole_id, sensor_id, part_name, str_time, resample_how):

    data_time_start = "{} {}".format(str_time, ' 00:00:00')
    data_time_end = "{} {}".format(str_time, ' 23:50:50')

    df_day = df[variable_all]
    df_day = df_day.loc[df_day['SENSOR_ID'] == sensor_id]
    df_day = df_day.loc[df_day['PART_NAME'] == part_name]
    # inplace True => 제자리에서 변경(새로운 DataFrame을 생성하지 않음)
    df_day.set_index('TIME_ID', inplace=True)
    df_day.index = pd.to_datetime(df_day.index)

    # 아웃라이어 제거
    for item in variable_plot:
        if item in limit_data.keys():
            mask = df_day[item] < limit_data[item][0]
            df_day.loc[mask, item] = None

            mask = df_day[item] > limit_data[item][1]
            df_day.loc[mask, item] = None

    # resample을 하기 위해서는 인덱스의 형식이 datetime 형식이어야 한다.
    df_day = df_day.resample(resample_how).mean()
    date_range = pd.date_range(data_time_start, data_time_end, freq=resample_how)
    df_day = df_day.reindex(date_range)

    # plot 할 때 모든 인덱스를 출력하기 위해 맨 처음 행과 맨 마지막 행에 값 삽입
    # 맨 처음 행과 맨 마지막 행에 값이 없을 경우에 삽입하며 ylim의 최솟값을 삽입
    # 꼼수이기 때문에 다른 방법이 있다면 바꿔야 한다.
    for idx in range(len(df_day.columns)):
        if pd.isnull(df_day.iloc[0, idx]):
            df_day.iloc[0, idx] = limit_ylim[df_day.columns[idx]][0]
        if pd.isnull(df_day.iloc[-1, idx]):
            df_day.iloc[-1, idx] = limit_ylim[df_day.columns[idx]][0]

    return df_day


def saveImage(df, dir_data, dir_output, pole_id, sensor_id, part_name, str_time, resample_how):

    start_time = time.time()

    file_name = '{}_{}_{}_{}'.format(pole_id ,sensor_id ,part_name ,str_time)
    df_day = getPole(df, pole_id, sensor_id, part_name, str_time, resample_how)

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(file_name)

    cnt = 0
    for item in variable_plot:
        cnt = cnt + 1
        exec('ax{} = fig.add_subplot({}, 1, {})'.format(str(cnt), str(len(variable_plot)), str(cnt)))
        eval('ax{}.plot(df_day[\'{}\'])'.format(str(cnt), item))
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


def sumDay(df):
    df.set_index(pd.to_datetime(df['Unnamed: 0']), inplace=True)
    df = df.resample("1D").sum()
    return df

def wrapper_saveImage(args):
    return saveImage(*args)

def main(path_dir, resample_how):

    # 데이터 디렉토리 체크
    dir_data = '{}data\\'.format(path_dir)
    if not os.path.isdir(dir_data):
        print('데이터 디렉토리가 존재하지 않습니다.')
        return

    # 아웃풋 디렉토리 체크
    dir_output = '{}output\\'.format(path_dir)
    if not os.path.isdir(dir_output):
        os.mkdir(dir_output)

    df_pole_50 = pd.read_csv('{}iot_pole_first_50.csv'.format(path_dir), encoding = "euc-kr")
    df_pole_part = pd.read_csv('{}iot_pole_1st_pole_part.csv'.format(path_dir), encoding = "euc-kr")
    df_10minute = pd.read_csv('{}1차폴의센서별10분당데이터유무.csv'.format(path_dir), encoding = "euc-kr")
    df_1day = sumDay(df_10minute)
    df_pole_part = df_pole_part.loc[df_pole_part['POLE_ID'].isin(df_pole_50['POLE_ID'].values)]


    for idx in df_pole_part.index.values:
        pole_id = df_pole_part['POLE_ID'][idx]
        sensor_id = df_pole_part['SENSOR_ID'][idx]
        part_name = df_pole_part['PART_NAME'][idx]

        if not os.path.exists('{}{}.csv'.format(dir_data, pole_id)):
            print('pass pole_id:{}'.format(pole_id))
            file_log = '{}file_log.txt'.format(dir_data)
            if not os.path.exists(file_log):
                os.mkdir(file_log)

            with open('{}새파일.txt'.format(dir_data), 'w') as f:
                data = 'pass pole_id:{}'.format(pole_id)
                f.write(data)
            continue

        df_pole = pd.read_csv('{}{}.csv'.format(dir_data, pole_id))

        df_1day_pole = df_1day.loc[df_1day['{}_{}_{}'.format(pole_id , sensor_id , part_name)] == 144, '{}_{}_{}'.format(pole_id , sensor_id , part_name)]

        cnt = 0
        list_pole = []
        for item in df_1day_pole.index.values:
            cnt = cnt + 1
            list_param = []

            list_param.append(df_pole)
            list_param.append(dir_data)
            list_param.append(dir_output)
            list_param.append(pole_id)
            list_param.append(sensor_id)
            list_param.append(part_name)
            list_param.append(str(item)[:10])
            list_param.append(resample_how)

            list_pole.append(list_param)

        print('create list end')

        with Pool(processes=5) as pool:
            pool.map(wrapper_saveImage, list_pole)

if __name__ == '__main__':

    start_time_main = time.time()

    path_dir = 'F:\\IOT\\'
    time_start = '2016-04-01 00:00:00'
    time_end = '2017-05-15 23:59:59'
    resample_how = '1T'

    main(path_dir, resample_how)

    print('Total Time:{}'.format(str(round(time.time() - start_time_main, 4))))

