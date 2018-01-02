import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import pandas as pd
import os, sys

plt.rc('font', family='HYsanB')


def getDataDay(pole_id, sensor_id, time_id):
    os.chdir('D:\\dev\\py_data\\pole_id_data2')
    data = pd.read_csv(pole_id + '.csv')
    # sensor = data['sensor_id'].unique()
    # sensor_num = data['sensor_id'].nunique()

    # time_id를 포함하는 하루 data
    # data_time = data[data['time_id'].str.contains(time_id)]
    # data_temp = data_time.loc[data['sensor_id'] == sensor_id, ['time_id', 'temp']]
    # data_temp['time_id'] = pd.to_datetime(data_temp['time_id'], format='%Y-%m-%d %H:%M:%S')
    # data_temp.set_index(data_temp['time_id'], inplace=True)  # set index as timeIndex
    # data_temp = data_temp.drop('time_id', 1)  # delete time column
    # data_temp.index.names = [None]

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
    # data_temp.to_csv(pole_id+'_'+time_id+'.csv')
    # print(data_temp)
    return data_temp


# pole = '8132W212'
# time='2016-10-09'
#
# sensor1 = '변압기 본체'
# tmp = getDataDay(pole, sensor1, time)
# tmp = tmp.resample('3T').max()
#
# sensor2 = '전주'
# tmp2 = getDataDay(pole, sensor2, time)
# tmp2 = tmp2.resample('3T').max()
#
# # xi=tmp.shape[0]
# # xi=list(range(xi))
# # x=tmp.index.format()
# # x=list(map(lambda x:x[14:16],x))
# temp=abs(tmp['temp']-tmp2['temp'])
#
# fig = plt.figure(figsize=(20,8))
# fig.suptitle(pole + '_' + time)
# ax = fig.add_subplot(211)
# ax.plot(tmp['temp'], label=sensor1)
# ax.plot(tmp2['temp'], label=sensor2)
# ax.legend(loc='upper right')
# ax2=fig.add_subplot(212)
# ax2.plot(temp, label=sensor1+' - '+sensor2)
# ax2.legend(loc='upper right')
# # plt.xticks(xi,x)
# plt.tight_layout()
# # fig.savefig('D:\\dev\\py_data\\result\\pole_temp\\temp_1026\\' + pole + '_' + time + '.png', format='png')
# plt.show()



data_list = [['8132W231', '2016-08-20'],
             ['8132W231', '2016-10-18'],
             # ['8132W231', '2017-04-21'],
             ['8132W231', '2017-04-22'],
             ['8132W231', '2017-04-28'],
             # ['8132W811', '2016-08-07'],
             ['8132W212', '2017-05-06'],
             ['8132W811', '2016-08-21'],
             ['8132W811', '2016-09-01']
             # ['8132W811', '2016-09-23'],
             # ['8132W212', '2017-04-23'],
             # ['8132W811', '2016-11-01'],
             # ['8132Z092', '2016-08-07'],
             # ['8132Z092', '2016-08-29'],
             # ['8132Z092', '2017-05-04']
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
    temp.to_csv(pole+'_'+time+'.csv')
    # print(type(temp))
    # print(temp.tolist())
    # fig.suptitle(pole + '_' + time)



    ax.plot(temp.tolist(), label=pole + '_' + time)
    # ax.legend(loc='upper right')

    # ax2 = fig.add_subplot(212)
    # ax2.plot(temp, label=sensor1 + ' - ' + sensor2)
    # # ax2.set_yticks(np.linspace(-12, 12, 13, endpoint=True))
    # ax2.legend(loc='upper right')
    # ax2.grid()
    # plt.tight_layout()
    # fig.savefig('D:\\dev\\py_data\\result\\pole_temp\\temp_1026_4\\' + pole + '_' + time + '.png', format='png')
    # plt.show()

ax.set_xticks(list_cnt)
ax.set_xticklabels(list_time, rotation='vertical')
ax.legend()
ax.grid()

plt.show()
