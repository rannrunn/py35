# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
import csv


dir = 'C:\\_data\\output_db_file_pi'

list_files = []
for path, dirs, files in os.walk(dir):
    # print(path)
    list_files = files
    break

print(list_files)

list_result_1 = []
list_result_2 = []


print(list_result_1)
path_file = 'C:\\_data\\time_diff.txt'

dir = 'C:\\_data\\output_db_file_pi'
list_min = []
list_xx = []
pole_id = ''
sensor_oid = list_files[0][list_files[0].rfind('_') + 1:list_files[0].rfind('.')]
df = pd.read_csv(os.path.join(dir, list_files[0]), encoding='euckr')
df['TIME_ID'] = pd.to_datetime(df['TIME_ID'])
df['TIME'] = df['TIME_ID']
df.set_index('TIME_ID', inplace=True)
df = df.drop(df[df['TEMP'].isnull()].index)
print(df[df['TEMP'].isnull()])
df['TIME_DIFF'] = df['TIME'].diff()
min = df['TIME_DIFF'].min().total_seconds()
# print(df[df['TIME_DIFF'].dt.total_seconds() == 0][-1:])
# print(min)
# print(sensor_oid)
if 180 > min:
    list_min.append(tuple([sensor_oid, df[df['TIME_DIFF'].dt.total_seconds() == 0][-1:]['TIME'].values]))


#Assuming res is a flat list
with open(path_file, "w") as output:
    for val in list_result_1:
        output.write(val + '\n')










