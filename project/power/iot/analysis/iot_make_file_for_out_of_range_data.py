# coding: utf-8

import pandas as pd
import os
from multiprocessing import Pool
import time
import numpy as np

def remove_out_of_range_data(df):
    if 'TEMP' in df.columns:
        df['TEMP'] = df['TEMP'].apply(lambda x: x if -40 <= x and x <= 125 else np.nan)
    if 'HUMI' in df.columns:
        df['HUMI'] = df['HUMI'].apply(lambda x: x if 0 <= x and x <= 100 else np.nan)
    if 'PITCH' in df.columns:
        df['PITCH'] = df['PITCH'].apply(lambda x: x if -180 <= x and x <= 180 else np.nan)
    if 'ROLL' in df.columns:
        df['ROLL'] = df['ROLL'].apply(lambda x: x if -90 <= x and x <= 90 else np.nan)
    if 'AMBIENT' in df.columns:
        df['AMBIENT'] = df['AMBIENT'].apply(lambda x: x if 0 <= x and x <= 60000 else np.nan)
    if 'UV' in df.columns:
        df['UV'] = df['UV'].apply(lambda x: x if 0 <= x and x <= 20.48 else np.nan)
    if 'BATTERY' in df.columns:
        df['BATTERY'] = df['BATTERY'].apply(lambda x: x if 0 <= x and x <= 100 else np.nan)
    if 'GEOMAG_X' in df.columns:
        df['GEOMAG_X'] = df['GEOMAG_X'].apply(lambda x: x if -5000 <= x and x <= 5000 else np.nan)
    if 'GEOMAG_Y' in df.columns:
        df['GEOMAG_Y'] = df['GEOMAG_Y'].apply(lambda x: x if -5000 <= x and x <= 5000 else np.nan)
    if 'GEOMAG_Z' in df.columns:
        df['GEOMAG_Z'] = df['GEOMAG_Z'].apply(lambda x: x if -5000 <= x and x <= 5000 else np.nan)
    if 'VAR_X' in df.columns:
        df['VAR_X'] = df['VAR_X'].apply(lambda x: x if -16000 <= x and x <= 16000 else np.nan)
    if 'VAR_Y' in df.columns:
        df['VAR_Y'] = df['VAR_Y'].apply(lambda x: x if -16000 <= x and x <= 16000 else np.nan)
    if 'VAR_Z' in df.columns:
        df['VAR_Z'] = df['VAR_Z'].apply(lambda x: x if -16000 <= x and x <= 16000 else np.nan)
    if 'USN' in df.columns:
        df['USN'] = df['USN'].apply(lambda x: x if 0 <= x and x <= 3000 else np.nan)
    if 'NTC' in df.columns:
        df['NTC'] = df['NTC'].apply(lambda x: x if -20 <= x and x <= 120 else np.nan)
    if 'UVC' in df.columns:
        df['UVC'] = df['UVC'].apply(lambda x: x if 0 <= x and x <= 5 else np.nan)

    return df

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

    df = df[(df['TEMP'] < -40) | (df['TEMP'] > 125)
            | (df['HUMI'] < 0) | (df['HUMI'] > 100)
            | (df['PITCH'] < -180) | (df['PITCH'] > 180)
            | (df['ROLL'] < -90) | (df['ROLL'] > 90)
            | (df['AMBIENT'] < 0) | (df['AMBIENT'] > 82000)
            | (df['UV'] < 0) | (df['UV'] > 20.48)
            | (df['PRESS'] < 300) | (df['PRESS'] > 1100)
            | (df['BATTERY'] < 0) | (df['BATTERY'] > 100)
            | (df['GEOMAG_X'] < -5000) | (df['GEOMAG_X'] > 5000)
            | (df['GEOMAG_Y'] < -5000) | (df['GEOMAG_Y'] > 5000)
            | (df['GEOMAG_Z'] < -5000) | (df['GEOMAG_Z'] > 5000)
            | (df['VAR_X'] < -16000) | (df['VAR_X'] > 16000)
            | (df['VAR_Y'] < -16000) | (df['VAR_Y'] > 16000)
            | (df['VAR_Z'] < -16000) | (df['VAR_Z'] > 16000)
            | (df['USN'] < 0) | (df['USN'] > 3000)
            | (df['NTC'] < -20) | (df['NTC'] > 120)
            | (df['UVC'] < 0) | (df['UVC'] > 5)
    ]
    df = df.dropna(axis=1, how='all')

    print(pole_id + '_' + df.index.name)

    return df


# resample 을 하기 전에 오류 값을 제거해야 함
# TEMP,HUMI,PITCH,ROLL,AMBIENT,UV,PRESS,BATTERY,PERIOD,CURRENT,SHOCK,GEOMAG_X,GEOMAG_Y,GEOMAG_Z,VAR_X,VAR_Y,VAR_Z,USN,NTC,UVC
if __name__ == '__main__':

    start = time.time()

    plot_type = 'data_exist'

    dir = 'C:/_data/output_db_file_pi'

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
    cmt_image = 50
    for idx in range(0, len(list_file_path), cmt_image):
        list_file_path_part = list_file_path[idx:idx + (cmt_image - 1)]

        with Pool(processes=16) as pool:
            list_df = pool.map(make_df, list_file_path_part)
            pool.close()
            pool.join()

        df_result = pd.concat([df_result, *list_df])

        for item_df in list_df:
            cnt_df = cnt_df + len(item_df)

    list_possible_col = ['POLE_ID', 'SENSOR_OID', 'TEMP', 'HUMI', 'PITCH', 'ROLL', 'AMBIENT', 'UV', 'PRESS', 'BATTERY', 'GEOMAG_X', 'GEOMAG_Y', 'GEOMAG_Z', 'VAR_X', 'VAR_Y', 'VAR_Z', 'USN', 'NTC', 'UVC']
    list_col = [item for item in list_possible_col if item in df_result.columns]

    df_result = df_result[list_col]
    df_result.to_csv(os.path.join('C:\\_data\\output', 'out_of_range_data.csv'))

    print('Total cnt:{}'.format(cnt_df))
    print('Total Time:{}'.format(str(round(time.time() - start))))



