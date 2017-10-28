import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import pandas as pd
import os, sys
import csv

plt.rc('font', family='HYsanB')


def getDataDay(pole_id, sensor_id, time_id):
    os.chdir('G:/pole_id_data_all')
    data = pd.read_csv(pole_id + '.csv')

    data_sensor = data.loc[data['sensor_id'] == sensor_id, ['time_id', 'temp']]
    data_sensor = data_sensor[data_sensor['time_id'].str.contains(time_id)]
    data_sensor['time_id'] = pd.to_datetime(data_sensor['time_id'], format='%Y-%m-%d %H:%M:%S')

    # 시작시간 끝시간
    start_time = pd.to_datetime(time_id, format='%Y-%m-%d %H:%M:%S')
    end_time = start_time.replace(hour=23, minute=59, second=59)
    time_range = pd.date_range(start_time, end_time, freq='s')
    data_time = pd.DataFrame(time_range)
    data_time.rename(columns={0: 'time_id'}, inplace=True)
    data_temp = pd.merge(data_time, data_sensor, on='time_id', how='left')
    data_temp['time_id'] = pd.to_datetime(data_temp['time_id'], format='%Y-%m-%d %H:%M:%S')
    # data_temp['time_id'] = pd.to_datetime(data_temp['time_id'],format='%H:%M').dt.time
    data_temp.set_index(data_temp['time_id'], inplace=True)  # set index as timeIndex
    data_temp = data_temp.drop('time_id', 1)  # delete time column
    data_temp.index.names = [None]
    # print(data_temp)
    return data_temp


data_list = [
    ['8132W212','2016-09-17'],
    # ['8132W231','2016-09-17'],
    # ['8132Z092','2016-09-17'],
    # ['8132X122','2016-09-17'],
    # ['8132W133','2016-09-17'],
    # ['8132W621','2016-09-17'],
    # ['8132X201','2016-09-17'],
    ['8132W811','2016-09-17'],
    ['8132W611','2016-09-17'],
    ['8132W821','2016-09-17'],
    # ['8232R131','2016-09-17'],
    # ['8132X782','2016-09-17'],
    # ['8132Z815','2016-09-17'],
    ['8132X021','2016-09-17'],
    # ['8132Z762','2016-09-17'],
    # ['8132W832','2016-09-17'],
    ['8132X474','2016-09-17'],
    # ['8132W081','2016-09-17'],
    ['8132X502','2016-09-17']
             ]


FMT = '%H:%M:%S'
x_start = '00:00:00'
x_time = x_start
list = [x_start[:-3]]
for i in range(143):
    x_time = (datetime.datetime.strptime(x_time, FMT) + datetime.timedelta(minutes=10)).strftime(FMT)
    list.append((x_time[:-3]))

list_cnt = np.arange(0,143,6)
list_time = []
for cnt in list_cnt:
    list_time.append(list[cnt])



fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111)

for pole, time in data_list:
    sensor1 = '변압기 본체'
    tmp = getDataDay(pole, sensor1, time)
    tmp_1 = tmp.resample('10T').max()
    # tmp = tmp.resample('3T').max()

    sensor2 = '전주'
    tmp2 = getDataDay(pole, sensor2, time)
    tmp2_1 = tmp2.resample('10T').max()
    # tmp2 = tmp2.resample('3T').max()

    temp = tmp_1['temp'] - tmp2_1['temp']
    print(type(temp))
    temp.to_csv('data_'+pole+'.csv')
    # print(type(temp))
    # print(temp.tolist())
    # fig.suptitle(pole + '_' + time)

    ax.plot(temp.tolist(), label=pole + '_' + time)

ax.set_xticks(list_cnt)
ax.set_xticklabels(list_time, rotation='vertical')
ax.legend()
ax.grid()

plt.show()
