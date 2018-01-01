import matplotlib.pyplot as plt
import pandas as pd
import traceback
import time
import os
import MySQLdb
import numpy as np


if __name__ == '__main__':

    start_time_main = time.time()

    # DB connection
    con = MySQLdb.connect('localhost', 'root', '1111', 'kepco')
    con.set_character_set('utf8')
    cur = con.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SET NAMES utf8;')
    cur.execute('SET CHARACTER SET utf8;')
    cur.execute('SET character_set_connection=utf8;')

    time_start = '2017-04-22 00:00:00'
    time_end = '2017-12-27 23:59:59'
    resample_how = '10T'


    df = pd.DataFrame()
    date_range = pd.date_range('2017-01-01 00:00:00', periods=2, freq='1T')
    df = df.reindex(date_range)
    df['TEMP'] = '22'
    df['TEMP'][0] = None
    print(df)
    print(df['TEMP'].isnull())
    print(df.loc[ df['TEMP'].isnull(), 'TEMP'].index)



    print('Total Time:' + str(round(time.time() - start_time_main, 4)))


