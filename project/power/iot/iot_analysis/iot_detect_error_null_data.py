# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from multiprocessing import Pool


def get_list_null_data(file_name):
    dir = 'C:\\_data\\output_db_file_pi'

    sensor_oid = file_name[file_name.rfind('_') + 1:file_name.rfind('.')]
    df = pd.read_csv(os.path.join(dir, file_name), encoding='euckr')
    df['TIME_ID'] = pd.to_datetime(df['TIME_ID'])
    df['TIME'] = df['TIME_ID']
    df.set_index('TIME_ID', inplace=True)
    df = df[df['TEMP'].isnull()]
    if len(df) > 0 :
        result = tuple([sensor_oid, df['TIME'].values[-1:][0]])
        return result


if __name__ == '__main__':

    dir = 'C:\\_data\\output_db_file_pi'

    list_files = []
    for path, dirs, files in os.walk(dir):
        # print(path)
        list_files = files
        break

    print(list_files)

    with Pool(processes=16) as pool:
        list_pool = pool.map(get_list_null_data, list_files)
        pool.close()

    path_file = 'C:\\_data\\iot_null_data.txt'

    if not os.path.isdir(path_file):
        os.makedirs(os.path.dirname(path_file))

    #Assuming res is a flat list
    with open(path_file, "w") as output:
        for item in list_pool:
            if item != None:
                output.write(str(item[0]) + ', ' + str(item[1]) + '\n')




