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
    if period_image == 'a':
        # 무조건 이미지 저장
        saveImage(df, dir_output, period_image)
    # 월별 데이터
    elif period_image == 'm':
        range_start = datetime.datetime(index_start.year, index_start.month, 1)
        range_end = range_start + relativedelta(months=1) - datetime.timedelta(seconds=1)
        while range_start < index_end:
            df_data = df.loc[range_start:range_end]
            # 데이터가 한개라도 있을 경우 이미지 저장
            if len(df_data.loc[df_data['TEMP'].notnull()]) != 0:
                saveImage(df_data, dir_output, period_image)
            range_start += relativedelta(months=1)
            range_end = range_start + relativedelta(months=1) - datetime.timedelta(seconds=1)
    # 주별 데이터
    elif period_image == 'w':
        step_week = step_day * 7
        range_start = datetime.datetime(index_start.year, index_start.month, index_start.day)
        # 일요일
        while range_start.weekday() != 6:
            range_start += datetime.timedelta(1)
        range_end = range_start + datetime.timedelta(7) - datetime.timedelta(seconds=1)
        while range_start < index_end:
            df_data = df[range_start:range_end]
            # 데이터가 모두 있을 경우 이미지 저장
            if len(df_data[df_data.iloc[:, 0].notnull()]) == step_week:
                saveImage(df_data, dir_output, period_image)
            range_start += datetime.timedelta(days=7)
            range_end += datetime.timedelta(days=7)
    # 일별 데이터
    elif period_image == 'd':
        range_start = datetime.datetime(index_start.year, index_start.month, index_start.day)
        range_end = range_start + datetime.timedelta(1) - datetime.timedelta(seconds=1)
        while range_start < index_end:
            df_data = df[range_start:range_end]
            # 데이터가 모두 있을 경우 이미지 저장
            if len(df_data[df_data.iloc[:, 0].notnull()]) == step_day:
                saveImage(df_data, dir_output, period_image)
            range_start += datetime.timedelta(days=1)
            range_end += datetime.timedelta(days=1)

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


def saveImage(df, dir_output, period_image):

    index_start = datetime.datetime.strptime(str(df.index.values[0])[:19], '%Y-%m-%dT%H:%M:%S')
    index_end = datetime.datetime.strptime(str(df.index.values[-1])[:19], '%Y-%m-%dT%H:%M:%S')

    print(df)

    if period_image == 'a':
        file_name = '{}'.format('ALL')
    else:
        file_name = '{}'.format(str(df.index.values[0])[:10])


    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(file_name)
    # print(df[df['TEMP'].notnull()])
    # df = df.interpolate()

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(df)
    ax1.set_xlim([datetime.datetime(index_start.year, index_start.month, index_start.day, index_start.hour, index_start.minute, index_start.second), datetime.datetime(index_end.year, index_end.month, index_end.day, index_end.hour, index_end.minute, index_end.second)])
    ax1.set_ylim([-50, 150000])
    plt.grid()

    try:
        fig.savefig('{}{}.png'.format(dir_output, file_name), format='png')
        plt.close(fig)
    except Exception as ex:
        traceback.print_exc()
        print('Exception:{}'.format(file_name))
    finally:
        plt.close(fig)

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

