import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir('D:\\dev\\IoT_data\\IoT_233')
data = pd.read_csv('8132D823.csv')
start = '2016-04-01 00:00:00'
end = '2017-05-15 23:59:59'

pole_data = data[['TIME_ID', 'AMBIENT', 'BATTERY', 'HUMI', 'PITCH', 'ROLL', 'PRESS', 'TEMP', 'UV']]
pole_data['TIME_ID'] = pd.to_datetime(pole_data['TIME_ID'], format='%Y-%m-%d %H:%M:%S')
print(pole_data)

# data의 주기가 일정하지 않아 초 단위로 time_id 컬럼을 가진 data_time 생성
start_time = pd.to_datetime(start, format='%Y-%m-%d %H:%M:%S')
end_time = pd.to_datetime(end, format='%Y-%m-%d %H:%M:%S')
time_range = pd.date_range(start_time, end_time, freq='s')
data_time = pd.DataFrame(time_range)
data_time.rename(columns={0: 'TIME_ID'}, inplace=True)  # inplace=True: data_time을 직접 변경

# time_id를 기준으로 data_time과 data_sensor merge
data_temp = pd.merge(data_time, pole_data, on='TIME_ID', how='left')
data_temp['TIME_ID'] = pd.to_datetime(data_temp['TIME_ID'], format='%Y-%m-%d %H:%M:%S')
# index를 time_id로 지정
data_temp.set_index(data_temp['TIME_ID'], inplace=True)
data_temp = data_temp.drop('TIME_ID', 1)
data_temp.index.names = [None]

print(data_temp)