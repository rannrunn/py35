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
def select(table, cur, pole_id):
    query = """SELECT DATE_FORMAT(CONCAT(CONVERT(COLUMN_MINUTE, CHAR(16)), '00'), '%%Y-%%m-%%d %%H:%%i') AS COLUMN_MINUTE,POLE_ID,BB_AVG_TEMP,BG_AVG_TEMP,WG_AVG_TEMP,JJ_AVG_TEMP,TH_AVG_TEMP 
                FROM TB_IOT_POLE_MINUTE_AVG_TEMP 
                WHERE POLE_ID = '%s' AND COLUMN_MINUTE BETWEEN '201608050000' AND '201608052359' 
                ORDER BY COLUMN_MINUTE""" % (pole_id)
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

    list_key = ['COLUMN_MINUTE','POLE_ID','BB_AVG_TEMP','BG_AVG_TEMP','WG_AVG_TEMP','JJ_AVG_TEMP','TH_AVG_TEMP']

    pole_id = '8132W212'
    table = 'TB_IOT_POLE_MINUTE_AVG_TEMP'

    list_sel = select(table, cur, pole_id)

    df = pd.DataFrame(list_sel)
    COLUMN_MINUTE = df.COLUMN_MINUTE.values
    BB_AVG_TEMP = df.BB_AVG_TEMP.values
    BG_AVG_TEMP = df.BG_AVG_TEMP.values
    WG_AVG_TEMP = df.WG_AVG_TEMP.values
    JJ_AVG_TEMP = df.JJ_AVG_TEMP.values
    TH_AVG_TEMP = df.TH_AVG_TEMP.values

    print(COLUMN_MINUTE)
    x_sequence = [i for i in range(len(COLUMN_MINUTE))]
    print(x_sequence)
    print(len(x_sequence))
    print(type(BB_AVG_TEMP))
    print(BB_AVG_TEMP.shape)

    x_arange_top = make_x_arange(x_sequence, 180)
    x_ticks_top = make_x_tick(COLUMN_MINUTE, x_arange_top)
    fig = plt.figure(figsize=(15, 13))
    ax1 = fig.add_subplot(10, 1, 1)
    ax2 = fig.add_subplot(10, 1, 2)
    ax3 = fig.add_subplot(10, 1, 3)
    ax4 = fig.add_subplot(10, 1, 4)
    ax5 = fig.add_subplot(10, 1, 5)
    ax6 = fig.add_subplot(2, 1, 2)
    fig.suptitle('xx')
    # axes, axes_name, axes_data, x_sequence, x_arange, x_ticks, axes_xlim, axes_ylim
    set_subplot(ax1, '변압기 본체',BB_AVG_TEMP, x_sequence, x_arange_top, ['' for i in range(len(x_arange_top))], -20, 55)
    set_subplot(ax2, '부하 개폐기',BG_AVG_TEMP, x_sequence, x_arange_top, ['' for i in range(len(x_arange_top))], -20, 55)
    set_subplot(ax3, '완금',WG_AVG_TEMP, x_sequence, x_arange_top, ['' for i in range(len(x_arange_top))], -20, 55)
    set_subplot(ax4, '전주',JJ_AVG_TEMP, x_sequence, x_arange_top, ['' for i in range(len(x_arange_top))], -20, 55)
    set_subplot(ax5, '통신용 함체',TH_AVG_TEMP, x_sequence, x_arange_top, x_ticks_top, -20, 55)

    print(BG_AVG_TEMP)

    x_arange_bottom = make_x_arange(x_sequence, 60)
    x_ticks_bottom = make_x_tick(COLUMN_MINUTE, x_arange_bottom)
    plt.plot(BB_AVG_TEMP, label='변압기 본체')
    # ax6.plot(x_time, BG_AVG_TEMP, label='부하 개폐기')
    # ax6.plot(x_time, WG_AVG_TEMP, label='완금')
    plt.plot(JJ_AVG_TEMP, label='전주')
    # ax6.plot(x_time, TH_AVG_TEMP, label='통신용 함체')
    plt.xticks(x_arange_bottom, x_ticks_bottom, rotation='vertical')
    plt.legend()


# axes.plot(x_sequence, axes_data)
# axes.set_xticks(x_arange)
# axes.set_xticklabels(x_ticks)
# axes.set_ylabel(axes_name)
# axes.set_ylim( [axes_xlim, axes_ylim] )

    print('past time:', time.time() - start)

    plt.show()



