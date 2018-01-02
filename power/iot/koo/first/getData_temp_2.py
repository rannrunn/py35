import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import pandas as pd
import os, sys
import multiprocessing as mp

os.chdir('G:/pole_id_data_all')
data = pd.read_csv('8132W122.csv')
sensor = data['sensor_id'].unique()
sensor_num = data['sensor_id'].nunique()
print(sensor)
print(sensor_num)
data_1 = data.loc[data['sensor_id'] == sensor[0], ['temp', 'sensor_id', 'time_id']]
data_2 = data.loc[data['sensor_id'] == sensor[1], ['temp', 'sensor_id', 'time_id']]
data_3 = data.loc[data['sensor_id'] == sensor[2], ['temp', 'sensor_id', 'time_id']]

id_1 = data_1['sensor_id'].unique()[0]
id_2 = data_2['sensor_id'].unique()[0]
id_3 = data_3['sensor_id'].unique()[0]
data_1['time_id']=pd.to_datetime(data_1['time_id'],format='%Y-%m-%d %H:%M:%S')
data_1.set_index(data_1['time_id'], inplace=True)  # set index as timeIndex
data_1 = data_1.drop('time_id', 1)  # delete time column
data_1 = data_1.drop('sensor_id', 1)  # delete time column
data_1.index.names = [None]

data_2['time_id']=pd.to_datetime(data_2['time_id'],format='%Y-%m-%d %H:%M:%S')
data_2.set_index(data_2['time_id'], inplace=True)  # set index as timeIndex
data_2 = data_2.drop('time_id', 1)  # delete time column
data_2 = data_2.drop('sensor_id', 1)  # delete time column
data_2.index.names = [None]

data_3['time_id']=pd.to_datetime(data_3['time_id'],format='%Y-%m-%d %H:%M:%S')
data_3.set_index(data_3['time_id'], inplace=True)  # set index as timeIndex
data_3 = data_3.drop('time_id', 1)  # delete time column
data_3 = data_3.drop('sensor_id', 1)  # delete time column
data_3.index.names = [None]

# print(data_1[250000:260000])
# print(data_1['temp'])
fig = plt.figure(figsize=(15, 3))
ax = plt.subplot(111)
ax.plot(data_1['temp'], label=id_1)
ax.plot(data_2['temp'], label=id_2)
ax.plot(data_3['temp'], label=id_3)
# ax.set_xlim(dstart, dend)
ax.legend()
plt.show()