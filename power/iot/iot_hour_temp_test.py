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
    query = """SELECT DATE_FORMAT(CONCAT(CONVERT(COLUMN_MINUTE, CHAR(16)), '00'), '%%Y-%%m-%%d %%H:%%i:%%s') AS COLUMN_MINUTE,POLE_ID,BB_AVG_TEMP,BG_AVG_TEMP,WG_AVG_TEMP,JJ_AVG_TEMP,TH_AVG_TEMP 
                FROM TB_IOT_POLE_MINUTE_AVG_TEMP 
                WHERE POLE_ID = '%s' AND COLUMN_MINUTE BETWEEN '%s' AND '%s'
                ORDER BY COLUMN_MINUTE limit 100;""" % (pole_id, time_start, time_end)
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


    table = 'TB_IOT_POLE_MINUTE_AVG_TEMP'

    pole_id = '8132W133'
    time_date = '20161112'
    time_start = time_date + '0000'
    time_end = time_date + '2359'

    list_sel = select(table, cur, pole_id, time_start, time_end)
    print(len(list_sel))
    dateRange = pd.date_range('20161112', periods=100, freq='min')
    df = pd.DataFrame(list_sel, index=dateRange)

    #print(dateRange)
    #df.reindex(dateRange)
    #print(df)
    df = df.fillna(0)
    print(df)
    dfff = df.max()
    print(dfff)

    plt.plot(df.BB_AVG_TEMP)
    #BB_AVG_TEMP,BG_AVG_TEMP,WG_AVG_TEMP,JJ_AVG_TEMP,TH_AVG_TEMP
    # plt.figure(figsize=(15,5))
    # x_count = [i for i in range(len(df.BB_AVG_TEMP))]
    # plt.plot(x_count, df.BB_AVG_TEMP, 'b', [i for i in range(len(df.BB_AVG_TEMP))], df.JJ_AVG_TEMP, 'r')
    # plt.xticks(np.arange(min(x_count), max(x_count) + 1, 60), df.COLUMN_MINUTE, rotation='vertical')
    # #np.arange(min(x_date), max(x_date)+1, 50000)
    # plt.legend()
    # plt.title('Bad Desired comm5d in Finger', size=15)
    # plt.legend(['Desired', 'Actual'])
    # plt.xlabel('time (s)', size=15)
    # plt.ylabel('degree', size=15)
    # plt.show()
    # plt.axis([1.5, 3, -2, 25])
    # plt.grid(True)
    # plt.annotate('Bad Desired', xy=(1.88, 15), xytext=(2.3, 16), size=15, arrowprops=dict(facecolor='black', shrink=0.04))
    plt.show()






