# coding: utf-8

import pandas as pd
import os
from multiprocessing import Pool
import time

def make_df(file_path):

    file_name = os.path.basename(file_path)
    if file_name.find('_') + 1 == file_name.rfind('_'):
        pole_id = ''
    else:
        pole_id = file_name[file_name.find('_') + 1:file_name.rfind('_')]

    sensor_oid = file_name[file_name.rfind('_') + 1:file_name.rfind('.')]

    df = pd.read_csv(file_path, encoding='euckr')
    df.set_index('TIME_ID', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.index.rename(sensor_oid, inplace=True)

    df = df[(df['TEMP'] == 300) | (df['HUMI'] == 300) | (df['PITCH'] == 500) | (df['ROLL'] == 500) | (df['AMBIENT'] == 60000) | (df['UV'] == 200) | (df['PRESS'] == 100)]
    df = df.dropna(axis=1, how='all')

    print(pole_id + '_' + df.index.name)

    return df


# resample 을 하기 전에 오류 값을 제거해야 함
# TEMP,HUMI,PITCH,ROLL,AMBIENT,UV,PRESS,BATTERY,PERIOD,CURRENT,SHOCK,GEOMAG_X,GEOMAG_Y,GEOMAG_Z,VAR_X,VAR_Y,VAR_Z,USN,NTC,UVC
if __name__ == '__main__':

    start = time.time()

    plot_type = 'data_exist'

    dir = 'C:\\_data\\output_db_file_pi'

    df_result = pd.DataFrame()

    list_file_path = []
    cnt = 0
    for path, dirs, files in os.walk(dir):
        # print(path)
        for file in files:
            cnt += 1
            # print('FILE_NAME:', file)
            list_file_path.append(os.path.join(path, file))


    cnt_df = 0
    cmt_image = 100
    for idx in range(0, len(list_file_path), cmt_image):
        list_file_path_part = list_file_path[idx:idx + (cmt_image - 1)]

        with Pool(processes=16) as pool:
            list_df = pool.map(make_df, list_file_path_part)
            pool.close()
            pool.join()

        df_result = pd.concat([df_result, *list_df])

        for item_df in list_df:
            cnt_df = cnt_df + len(item_df)

    list_possible_col = ['POLE_ID', 'SENSOR_OID', 'TEMP', 'HUMI', 'PITCH', 'ROLL', 'AMBIENT', 'UV', 'PRESS', 'BATTERY', 'PERIOD']
    list_col = [item for item in list_possible_col if item in df_result.columns]

    df_result = df_result[list_col]
    df_result.to_csv(os.path.join('C:\\_data\\output', 'sensor_malfunction_data.csv'))

    print('Total cnt:{}'.format(cnt_df))
    print('Total Time:{}'.format(str(round(time.time() - start))))



