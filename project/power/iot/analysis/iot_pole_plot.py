import matplotlib.pyplot as plt
import pandas as pd
import os

import time

# information 파일 읽기
df_sensor_mounting_position = pd.read_csv('C:/_data/iot_sensor_mounting_position.csv', encoding='euckr')
df_location = pd.read_csv('C:/_data/iot_location.csv', encoding='euckr')

dir = 'D:/IoT/output_db_file_pi'




list_ansan_2 = []
list_daegu_1 = []
list_daegu_2 = []
list_gochang_2 = []
list_gwangju_1 = []
list_gwangju_2 = []

df_location['FILE_NAME'] = df_location['SENSOR_OID'].str.slice(0, 1) + '_' + df_location['POLE_CPTZ_NO'] + '_' + df_location['SENSOR_OID'] + '.csv'

print(df_location['FILE_NAME'])

mask_ansan_2 = df_location['POLE_ADDR'].str.contains('안산센서')
mask_daegu_1 = df_location['POLE_ADDR'].str.contains('대구 광역시')
mask_daegu_2 = df_location['POLE_ADDR'].str.contains('대구센서')
mask_gochang_2 = df_location['POLE_ADDR'].str.contains('고창군')
mask_gwangju_1 = df_location['POLE_ADDR'].str.contains('광주 광역시')
mask_gwangju_2 = df_location['POLE_ADDR'].str.contains('광주센서')


list_ansan_2 = df_location[mask_ansan_2]['FILE_NAME'].tolist()
list_daegu_1 = df_location[mask_daegu_1]['FILE_NAME'].tolist()
list_daegu_2 = df_location[mask_daegu_2]['FILE_NAME'].tolist()
list_gochang_2 = df_location[mask_gochang_2]['FILE_NAME'].tolist()
list_gwangju_1 = df_location[mask_gwangju_1]['FILE_NAME'].tolist()
list_gwangju_2 = df_location[mask_gwangju_2]['FILE_NAME'].tolist()
print(df_location[mask_daegu_2])
print(df_location[mask_daegu_1])

list_read_csv = list_daegu_1
cnt = 0
for path, dirs, files in os.walk(dir):
    print(path)
    for file in files:
        if file in list_read_csv:
            cnt += 1
            print('FILE_NAME:', file)
            df_file = pd.read_csv(os.path.join(path, file), encoding='euckr')


