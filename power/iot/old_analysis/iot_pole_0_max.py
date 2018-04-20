# coding: utf-8
import os
import time
import pandas as pd
from matplotlib import font_manager, rc

import MySQLdb

import copy

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# max
# 1차 'TEMP':'300.000', 'HUMI':'300.000', 'PITCH':'500.000', 'ROLL':'500.000', 'AMBIENT':'60000.000', 'UV':'200.000', 'PRESS':'1031.900', 'BATTERY':'100.000', 'SHOCK':'1.000', 'GEOMAG_X':'159.000', 'GEOMAG_Y':'250.000', 'GEOMAG_Z':'1064.000', 'VAR_X':'937.000', 'VAR_Y':'10594.000', 'VAR_Z':'1089.000', 'USN':'235.000', 'NTC':'34.000', 'UVC':'0.367'
# 2차 'TEMP':'300.000', 'HUMI':'300.000', 'PITCH':'500.000', 'ROLL':'500.000', 'AMBIENT':'60000.000', 'UV':'200.000', 'PRESS':'1056.000', 'BATTERY':'207.810', 'SHOCK':'0.000', 'GEOMAG_X':'4204.000', 'GEOMAG_Y':'4915.000', 'GEOMAG_Z':'3705.000', 'VAR_X':'23815.000', 'VAR_Y':'15751.000', 'VAR_Z':'21429.000', 'USN':'278.000', 'NTC':'120.000', 'UVC':'7.040'
# 에러 제거 후 최대값 : 1차 AMBIENT 31889, 2차 AMBIENT 4906
#
# min
# 1차 'TEMP':'-40.000', 'HUMI':'0.000', 'PITCH':'-180.100', 'ROLL':'-90.000', 'AMBIENT':'0.000', 'UV':'0.000', 'PRESS':'0.000', 'BATTERY':'0.000', 'SHOCK':'0.000', 'GEOMAG_X':'-126.000', 'GEOMAG_Y':'-552.000', 'GEOMAG_Z':'-1253.000', 'VAR_X':'-5438.000', 'VAR_Y':'-1101.000', 'VAR_Z':'-1113.000', 'USN':'0.000', 'NTC':'6.000', 'UVC':'0.000'
# 2차 'TEMP':'-40.000', 'HUMI':'0.000', 'PITCH':'-180.100', 'ROLL':'-90.000', 'AMBIENT':'0.000', 'UV':'0.000', 'PRESS':'0.000', 'BATTERY':'0.000', 'SHOCK':'0.000', 'GEOMAG_X':'-4422.000', 'GEOMAG_Y':'-1737.000', 'GEOMAG_Z':'-3476.000', 'VAR_X':'-17966.000', 'VAR_Y':'-24002.000', 'VAR_Z':'-16126.000', 'USN':'0.000', 'NTC':'-20.000', 'UVC':'0.000'
# Total Time:844.1308

# DB connection
con = MySQLdb.connect('localhost', 'root', '1111', 'kepco')
con.set_character_set('utf8')
cur = con.cursor(MySQLdb.cursors.DictCursor)
cur.execute('SET NAMES utf8;')
cur.execute('SET CHARACTER SET utf8;')
cur.execute('SET character_set_connection=utf8;')

var = {'TEMP':'',
'HUMI':'',
'PITCH':'',
'ROLL':'',
'AMBIENT':'',
'UV':'',
'PRESS':'',
'BATTERY':'',
'PERIOD':'',
'CURRENT':'',
'SHOCK':'',
'GEOMAG_X':'',
'GEOMAG_Y':'',
'GEOMAG_Z':'',
'VAR_X':'',
'VAR_Y':'',
'VAR_Z':'',
'USN':'',
'NTC':'',
'UVC':''}


# SELECT
def selectMax(variable, table, max_min):
    query = """SELECT %s(convert(%s, decimal(8,3))) as %s
                FROM TB_IOT_POLE_%s
                """ % (max_min, variable, variable, table)
    cur.execute(query);
    results = cur.fetchall()
    df = pd.DataFrame(list(results))
    return df[variable][0]

def main():

    max_min = 'min'
    print(max_min)

    var_1 = copy.deepcopy(var)

    for key in var.keys():
        var_1[key] = selectMax(key, 'first', max_min)

    print(var_1)

    var_2 = copy.deepcopy(var)

    for key in var.keys():
        var_2[key] = selectMax(key, 'second', max_min)

    print(var_2)


if __name__ == '__main__':

    start_time_main = time.time()

    main()

    print('Total Time:{}'.format(str(round(time.time() - start_time_main, 4))))

