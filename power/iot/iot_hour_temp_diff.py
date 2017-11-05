# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import MySQLdb
import time
import datetime
import numpy as np
import pandas as pd
import os, sys
import multiprocessing as mp

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# SELECT 하는 방법
def select(table, cur, pole_id, time_start, time_end):
    query = """SELECT DATE_FORMAT(CONCAT(CONVERT(COLUMN_MINUTE, CHAR(16)), '00'), '%%Y-%%m-%%d %%H:%%i') AS COLUMN_MINUTE,POLE_ID
                , (CASE WHEN BB_AVG_TEMP IS NOT NULL AND JJ_AVG_TEMP IS NOT NULL THEN BB_AVG_TEMP - JJ_AVG_TEMP ELSE 0 END) AS BB_JJ_DIFF
                FROM TB_IOT_POLE_MINUTE_AVG_TEMP 
                WHERE POLE_ID = '%s' AND COLUMN_MINUTE BETWEEN '%s' AND '%s' 
                ORDER BY COLUMN_MINUTE""" % (pole_id, time_start, time_end)
    print(query)
    cur.execute(query);
    results = cur.fetchall()
    list = []
    for row in results:
        list.append(row)

    return list

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
    axes.set_ylim( [axes_xlim, axes_ylim] )

def date_add(time, cnt):

    pass

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

    table = 'TB_IOT_POLE_MINUTE_AVG_TEMP'

    pole_id = '8132W133'
    time_date = '20161112'
    time_start = time_date + '0000'
    time_end = time_date + '2359'

    list_mon = select(table, cur, pole_id, time_start, time_end)
    # list_tue = select(table, cur, pole_id, time_start, time_end)
    # list_wed = select(table, cur, pole_id, time_start, time_end)
    # list_thu = select(table, cur, pole_id, time_start, time_end)
    # list_fri = select(table, cur, pole_id, time_start, time_end)
    # list_sat = select(table, cur, pole_id, time_start, time_end)
    # list_sun = select(table, cur, pole_id, time_start, time_end)

    df = pd.DataFrame(list_mon)
    COLUMN_MINUTE = df.COLUMN_MINUTE.values
    BB_JJ_DIFF = df.BB_JJ_DIFF.values


    x_sequence = [i for i in range(len(COLUMN_MINUTE))]
    print(COLUMN_MINUTE)
    print(x_sequence)
    print(type(BB_JJ_DIFF))

    x_arange_top = make_x_arange(x_sequence, 180)
    x_ticks_top = make_x_tick(COLUMN_MINUTE, x_arange_top)
    fig = plt.figure(figsize=(15, 13))
    ax1 = fig.add_subplot(1, 1, 1)
    fig.suptitle(pole_id)
    # axes, axes_name, axes_data, x_sequence, x_arange, x_ticks, axes_xlim, axes_ylim


    print('BB_JJ_DIFF:', BB_JJ_DIFF)

    myDatetime = datetime.datetime.strptime('20150415232338', '%Y%m%d%H%M%S')
    #print(myDatetime. )

    x_arange_bottom = make_x_arange(x_sequence, 60)
    x_ticks_bottom = make_x_tick(COLUMN_MINUTE, x_arange_bottom)
    ax1.plot(x_sequence, BB_JJ_DIFF, label='변압기 - 전주')
    ax1.set_xticks(x_arange_bottom)
    ax1.set_xticklabels(x_ticks_bottom, rotation='vertical')
    ax1.set_ylim( [-5, 15] )

    print('past time:', time.time() - start)
    plt.legend()
    plt.show()




