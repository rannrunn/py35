# 변압기 본체, 부하 개폐기, 완금, 전주, 통신용 함체의 온도를 각 각 plot
# 변압기, 전주, 변압기-전주 세 가지 온도를 하나의 차트에 plot
# coding: utf-8
import time
import datetime
import MySQLdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# SELECT 하는 방법
def select(cur, pole_id, time_start, time_end):
    query = """SELECT DATE_FORMAT(CONCAT(CONVERT(COLUMN_MINUTE, CHAR(16)), '00'), '%%Y-%%m-%%d %%H:%%i:%%s') AS COLUMN_MINUTE,POLE_ID,BB_AVG_TEMP,BG_AVG_TEMP,WG_AVG_TEMP,JJ_AVG_TEMP,TH_AVG_TEMP 
                FROM TB_IOT_POLE_MINUTE_AVG_TEMP 
                WHERE POLE_ID = '%s' AND COLUMN_MINUTE BETWEEN '%s' AND '%s'  
                ORDER BY COLUMN_MINUTE""" % (pole_id, time_start, time_end)
    cur.execute(query);
    results = cur.fetchall()
    df_query = pd.DataFrame(list(results))
    df_query = df_query.set_index(pd.to_datetime(df_query['COLUMN_MINUTE']))
    return df_query

# 차트의 x축 생성
def make_x_arange(x_src, interval):
    return np.arange(min(x_src), max(x_src), interval)

# 차트의 y축 생성
def make_x_tick(list_data, list_cnt):
    list = []
    for cnt in list_cnt:
        list.append(list_data[cnt])
    return list

# subplot 함수
def set_subplot(axes, axes_name, axes_data, x_sequence, x_arange, x_ticks, axes_xlim, axes_ylim):
    axes.plot(x_sequence, axes_data)
    axes.set_xticks(x_arange)
    axes.set_xticklabels(x_ticks)
    axes.set_ylabel(axes_name, fontsize=15)
    axes.set_ylim([axes_xlim, axes_ylim])
    axes.tick_params(labelsize=15)


if __name__ == '__main__':

    start = time.time()

    # DB connection
    con = MySQLdb.connect('localhost', 'root', '1111', 'kepco')
    con.set_character_set('utf8')
    cur = con.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SET NAMES utf8;')
    cur.execute('SET CHARACTER SET utf8;')
    cur.execute('SET character_set_connection=utf8;')

    # 변수 설정
    pole_id = '8132W133'
    time_date = '20161112'
    time_start = time_date + '0000'
    time_end = time_date + '2359'

    # 쿼리 SELECT
    df_query = select(cur, pole_id, time_start, time_end)
    df_query[['BB_AVG_TEMP','BG_AVG_TEMP','WG_AVG_TEMP','JJ_AVG_TEMP','TH_AVG_TEMP']] = df_query[['BB_AVG_TEMP','BG_AVG_TEMP','WG_AVG_TEMP','JJ_AVG_TEMP','TH_AVG_TEMP']].astype('float64')
    # 중간값 채우기
    df_query[['BB_AVG_TEMP','BG_AVG_TEMP','WG_AVG_TEMP','JJ_AVG_TEMP','TH_AVG_TEMP']].interpolate()
    # 데이터를 3분씩 묶음
    df_3t = df_query.resample('3T').min()
    # 변압기 본체와 전주의 차이를 구함
    df_diff = df_3t['BB_AVG_TEMP'] - df_3t['JJ_AVG_TEMP']

    # x축의 시퀀스
    x_sequence = [i for i in range(len(df_3t['COLUMN_MINUTE']))]


    x_arange = make_x_arange(x_sequence, 20)
    x_ticks_no = ['' for i in range(len(x_arange))]
    x_ticks = make_x_tick(df_3t['COLUMN_MINUTE'], x_arange)

    x_ticks = [item[11:16] for item in x_ticks]

    # 사이즈 15, 10의 figure 생성
    fig = plt.figure(figsize=(20, 30))

    # 6개 차트 생성
    ax1 = fig.add_subplot(5, 1, 1)
    ax2 = fig.add_subplot(5, 1, 2)
    ax3 = fig.add_subplot(5, 1, 3)
    ax4 = fig.add_subplot(5, 1, 4)
    ax5 = fig.add_subplot(5, 1, 5)

    # subtitle 을 pole_id로 설정
    fig.suptitle(pole_id, fontsize=20)

    # 다섯개 변수의 subplot를 생성
    # axes, axes_name, axes_data, x_sequence, x_arange, x_ticks, axes_xlim, axes_ylim
    set_subplot(ax1, '변압기 본체', df_3t['BB_AVG_TEMP'], x_sequence, x_arange, x_ticks_no, -20, 55)
    set_subplot(ax2, '부하 개폐기', df_3t['BG_AVG_TEMP'], x_sequence, x_arange, x_ticks_no, -20, 55)
    set_subplot(ax3, '완   금', df_3t['WG_AVG_TEMP'], x_sequence, x_arange, x_ticks_no, -20, 55)
    set_subplot(ax4, '전   주', df_3t['JJ_AVG_TEMP'], x_sequence, x_arange, x_ticks_no, -20, 55)
    set_subplot(ax5, '통신용 함체', df_3t['TH_AVG_TEMP'], x_sequence, x_arange, x_ticks, -20, 55)

    print('past time:', time.time() - start)

    # 라벨 표시
    # plot 보여주기
    plt.show()
    plt.close()

    # 사이즈 15, 10의 figure 생성
    fig = plt.figure(figsize=(20, 30))
    # subtitle 을 pole_id로 설정
    fig.suptitle(pole_id, fontsize=20)
    ax1 = fig.add_subplot(1, 1, 1)

    # 변압기 본체, 전주, 변압기 본체 - 전주를 표현하는 차트를 생성
    ax1.plot(x_sequence, df_3t['BB_AVG_TEMP'], label='변압기 본체')
    ax1.plot(x_sequence, df_3t['JJ_AVG_TEMP'], label='전   주')
    ax1.plot(x_sequence, df_diff, label='차이')
    ax1.set_ylabel('온   도', fontsize=15)
    ax1.set_xticks(x_arange)
    ax1.set_xticklabels(x_ticks)
    ax1.tick_params(labelsize=15)
    # 라벨 표시
    plt.legend(fontsize=15)
    # plot 보여주기
    plt.show()
    plt.close()



