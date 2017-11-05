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

def make_x_arange(x_src, interval):
    return np.arange(min(x_src), max(x_src)+1, interval)

def make_x_tick(list_data, list_cnt):
    list = []
    for cnt in list_cnt:
        list.append(list_data[cnt])
    return list

def set_subplot(axes, axes_name, axes_data, x_sequence, x_arange, x_ticks, axes_xlim, axes_ylim):
    axes.plot(x_sequence, axes_data)
    axes.set_xticks(x_arange)
    axes.set_xticklabels(x_ticks)
    axes.set_ylabel(axes_name)
    axes.set_ylim([axes_xlim, axes_ylim])

def saveimage(filename):
    pass

if __name__ == '__main__':

    start = time.time()

    con = MySQLdb.connect('localhost', 'root', '1111', 'kepco')
    con.set_character_set('utf8')
    cur = con.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SET NAMES utf8;')
    cur.execute('SET CHARACTER SET utf8;')
    cur.execute('SET character_set_connection=utf8;')

    pole_id = '8132W133'
    time_date = '20161112'
    time_start = time_date + '0000'
    time_end = time_date + '2359'


    df_query = select(cur, pole_id, time_start, time_end)
    df_query[['BB_AVG_TEMP','BG_AVG_TEMP','WG_AVG_TEMP','JJ_AVG_TEMP','TH_AVG_TEMP']] = df_query[['BB_AVG_TEMP','BG_AVG_TEMP','WG_AVG_TEMP','JJ_AVG_TEMP','TH_AVG_TEMP']].astype('float64')
    df_query[['BB_AVG_TEMP','BG_AVG_TEMP','WG_AVG_TEMP','JJ_AVG_TEMP','TH_AVG_TEMP']].interpolate()

    df_3t = df_query.resample('3T').max()
    df_diff = df_3t['BB_AVG_TEMP'] - df_3t['JJ_AVG_TEMP']

    x_sequence = [i for i in range(len(df_3t['COLUMN_MINUTE']))]

    x_arange = make_x_arange(x_sequence, 180)
    x_ticks_no = ['' for i in range(len(x_arange))]
    x_ticks = make_x_tick(df_3t['COLUMN_MINUTE'], x_arange)
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(10, 1, 1)
    ax2 = fig.add_subplot(10, 1, 2)
    ax3 = fig.add_subplot(10, 1, 3)
    ax4 = fig.add_subplot(10, 1, 4)
    ax5 = fig.add_subplot(10, 1, 5)
    ax6 = fig.add_subplot(2, 1, 2)
    fig.suptitle(pole_id)
    # axes, axes_name, axes_data, x_sequence, x_arange, x_ticks, axes_xlim, axes_ylim
    set_subplot(ax1, '변압기 본체', df_3t['BB_AVG_TEMP'], x_sequence, x_arange, x_ticks_no, -20, 55)
    set_subplot(ax2, '부하 개폐기', df_3t['BG_AVG_TEMP'], x_sequence, x_arange, x_ticks_no, -20, 55)
    set_subplot(ax3, '완금', df_3t['WG_AVG_TEMP'], x_sequence, x_arange, x_ticks_no, -20, 55)
    set_subplot(ax4, '전주', df_3t['JJ_AVG_TEMP'], x_sequence, x_arange, x_ticks_no, -20, 55)
    set_subplot(ax5, '통신용 함체', df_3t['TH_AVG_TEMP'], x_sequence, x_arange, x_ticks, -20, 55)

    ax6.plot(x_sequence, df_3t['BB_AVG_TEMP'], label='변압기 본체')
    ax6.plot(x_sequence, df_3t['JJ_AVG_TEMP'], label='전주')
    ax6.plot(x_sequence, df_diff, label='차이')
    ax6.set_xticks(x_arange)
    ax6.set_xticklabels(x_ticks)

    print('past time:', time.time() - start)
    plt.legend()
    plt.show()



