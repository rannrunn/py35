# coding: utf-8
import datetime
import os
import time
import traceback

import MySQLdb
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.relativedelta import relativedelta
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# DB connection
con = MySQLdb.connect('localhost', 'root', '1111', 'kepco')
con.set_character_set('utf8')
cur = con.cursor(MySQLdb.cursors.DictCursor)
cur.execute('SET NAMES utf8;')
cur.execute('SET CHARACTER SET utf8;')
cur.execute('SET character_set_connection=utf8;')

# SELECT
def select(cur, time_start, time_end):
    query = """SELECT TIME_ID, FGMWGS0_865$334$FRI$8004_F_CV
                FROM TB_WATER_QUANTITY
                """
    cur.execute(query);
    results = cur.fetchall()
    df = pd.DataFrame(list(results))
    return df

def process(dir_output, period_image, time_start, time_end, resample_how):

    df = getData(time_start, time_end, resample_how)

    # 데이터의 기간을 설정
    date_range = pd.date_range(time_start, time_end, freq=resample_how)
    df = df.reindex(date_range)
    step_day = getStepDay(resample_how)

    print(df)

    index_start = datetime.datetime.strptime(str(df.index.values[0])[:19], '%Y-%m-%dT%H:%M:%S')
    index_end = datetime.datetime.strptime(str(df.index.values[-1])[:19], '%Y-%m-%dT%H:%M:%S')
    # 옵션에 데이터 따라 이미지 저장
    # 모든 데이터


# resample_how에 따른 step을 가져온다.
def getStepDay(resample_how):
    return {
        '1T':1440
        ,'10T':144
        ,'60T':24
        ,'120T':12
        ,'1D':1
    }.get(resample_how, 0)

# 전주별 데이터를 가져온다.
def getData(time_start, time_end, resample_how):
    col = 'FGMWGS0_865$334$FRI$8004_F_CV'
    df_sel = select(cur, time_start, time_end)
    df_sel[col] = df_sel[col].astype(float)
    df_sel.set_index('TIME_ID', inplace=True)
    df_sel.index = pd.to_datetime(df_sel.index)
    df_sel = df_sel.resample(resample_how).sum()

    return df_sel


def main(path_dir, period_image, time_start, time_end, resample_how):

    # 아웃풋 디렉토리 체크
    dir_output = '{}output\\{}\\'.format(path_dir, resample_how)
    if not os.path.isdir(dir_output):
        os.mkdir(dir_output)

    process(dir_output, period_image, time_start, time_end, resample_how)


if __name__ == '__main__':

    start_time_main = time.time()

    period_image = 'd' # all, month, week, day 의 이니셜
    path_dir = 'G:\\water\\'

    time_start = '2017-12-01 00:00:00'
    time_end = '2017-12-31 23:59:59'

    resample_how = '60T'

    main(path_dir, period_image, time_start, time_end, resample_how)

    print('Total Time:{}'.format(str(round(time.time() - start_time_main, 4))))

