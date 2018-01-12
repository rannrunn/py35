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
    query = """SELECT *
                FROM TB_IOT_POLE_FIRST
                WHERE POLE_ID = '%s'
                """ % (pole_id)
    cur.execute(query);
    results = cur.fetchall()
    df = pd.DataFrame(list(results))
    return df

def saveData(pole_id):
    df = selectPoleSensor(pole_id)
    df = df[['FILE_NAME','TIME_ID','POLE_ID','SENSOR_ID','PART_NAME','RI','PI','TEMP','HUMI','PITCH','ROLL','AMBIENT','UV','PRESS','BATTERY','PERIOD','CURRENT','SHOCK','GEOMAG_X','GEOMAG_Y','GEOMAG_Z','VAR_X','VAR_Y','VAR_Z','USN','NTC','UVC']]
    df.columns = ['FILE_NAME','TIME_ID','POLE_ID','SENSOR_ID','PART_NAME','RI','PI','TEMP','HUMI','PITCH','ROLL','AMBIENT','UV','PRESS','BATTERY','PERIOD','CURRENT','SHOCK','GEOMAG_X','GEOMAG_Y','GEOMAG_Z','VAR_X','VAR_Y','VAR_Z','USN','NTC','UVC']
    df.to_csv('{}output\\{}.csv'.format(path_dir, pole_id))

def wrapper(args):
    return saveData(*args)

def main(path_dir):

    # 아웃풋 디렉토리 체크
    dir_output = '{}output\\'.format(path_dir)
    if not os.path.isdir(dir_output):
        os.mkdir(dir_output)
        print('아웃풋 디렉토리를 생성하였습니다.')

    df_pole_info = pd.read_csv('{}iot_pole_1st_info.csv'.format(path_dir), encoding = "euc-kr")
    df_unique = df_pole_info[~df_pole_info['POLE_ID'].isin(df_pole_info[df_pole_info['PART_NAME'].isnull()]['POLE_ID'])]['POLE_ID'].unique()

    list = []
    for pole_id in df_unique:
        list_param = []
        list_param.append(pole_id)
        list.append(list_param)

    with Pool(processes=3) as pool:
        pool.map(wrapper, list)


if __name__ == '__main__':

    start_time_main = time.time()

    main(path_dir)

    print('Total Time:{}'.format(str(round(time.time() - start_time_main, 4))))

