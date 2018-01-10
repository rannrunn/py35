# coding: utf-8
import os
import time
import pandas as pd
from matplotlib import font_manager, rc

import MySQLdb

from multiprocessing import Pool

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# DB connection
con = MySQLdb.connect('localhost', 'root', '1111', 'kepco')
con.set_character_set('utf8')
cur = con.cursor(MySQLdb.cursors.DictCursor)
cur.execute('SET NAMES utf8;')
cur.execute('SET CHARACTER SET utf8;')
cur.execute('SET character_set_connection=utf8;')

path_dir = 'F:\\IOT\\'

# SELECT
def selectPoleSensor(pole_id):
    query = """select s1.shock as SHOCK_SEL, s1.PITCH as PITCH_SEL, s1.roll as ROLL_SEL,s1.* from tb_iot_pole_second s1 , tb_iot_pole_info_dummy s2
                where 1 = 1
                and s1.POLE_ID = s2.POLE_ID
                and s1.SENSOR_ID = s2.SENSOR_ID
                and s2.PART_NAME = '변압기 본체'
                and s1.POLE_ID = '%s'
                and shock <> ''
                """ % (pole_id)
    cur.execute(query);
    results = cur.fetchall()
    df = pd.DataFrame(list(results))
    return df

def saveData(df_pole_info, pole_id):
    df = selectPoleSensor(pole_id)
    df = df.merge(df_pole_info, on='SENSOR_ID', how='left')
    df = df[['FILE_NAME','TIME_ID','AREA','POLE_ID_x','SENSOR_ID','PART_NAME','RI','PI','TEMP','HUMI','PITCH','ROLL','AMBIENT','UV','PRESS','BATTERY','PERIOD','CURRENT','SHOCK','GEOMAG_X','GEOMAG_Y','GEOMAG_Z','VAR_X','VAR_Y','VAR_Z','USN','NTC','UVC']]
    df.columns = ['FILE_NAME','TIME_ID','AREA','POLE_ID','SENSOR_ID','PART_NAME','RI','PI','TEMP','HUMI','PITCH','ROLL','AMBIENT','UV','PRESS','BATTERY','PERIOD','CURRENT','SHOCK','GEOMAG_X','GEOMAG_Y','GEOMAG_Z','VAR_X','VAR_Y','VAR_Z','USN','NTC','UVC']
    df.to_csv('{}output\\{}.csv'.format(path_dir, pole_id))

def wrapper(args):
    return saveData(*args)

def main(path_dir):

    # 아웃풋 디렉토리 체크
    dir_output = '{}output\\'.format(path_dir)
    if not os.path.isdir(dir_output):
        os.mkdir(dir_output)
        print('아웃풋 디렉토리를 생성하였습니다.')

    df_pole_info = pd.read_csv('{}iot_pole_2nd_info.csv'.format(path_dir), encoding = "euc-kr")
    df_pole_2nd_30 = pd.read_csv('{}iot_pole_2nd_30.csv'.format(path_dir), encoding = "euc-kr")
    ser_unique = df_pole_info[~df_pole_info['POLE_ID'].isin(df_pole_info[df_pole_info['PART_NAME'].isnull()]['POLE_ID'])]['POLE_ID'].unique()
    df_pole_info = df_pole_info[df_pole_info['POLE_ID'].isin(ser_unique)]
    # df_pole_info = df_pole_info[df_pole_info['POLE_ID'].isin(df_pole_2nd_30['POLE_ID'])]

    cnt = 0
    for idx in df_pole_info.index.values:
        cnt += 1
        print(cnt)

        pole_id = df_pole_info['POLE_ID'][idx]
        sensor_id = df_pole_info['SENSOR_ID'][idx]

        df_pole = pd.read_csv('{}data\\{}.csv'.format(path_dir, pole_id), encoding = "euc-kr")

        df_pole = df_pole[df_pole['SHOCK'].notnull()]
        df_pole = df_pole[df_pole['SHOCK'] != 0]

        if len(df_pole) > 0:
            print(df_pole)
            print('shock pole_id:{}'.format(pole_id))
            print('shock sensor_id:{}'.format(sensor_id))

if __name__ == '__main__':

    start_time_main = time.time()

    main(path_dir)

    print('Total Time:{}'.format(str(round(time.time() - start_time_main, 4))))

