import MySQLdb
import numpy as np
import pandas as pd
from numpy import*

import common as comm
import dbconnection as conn


# {'location_target': 'd.paldang + d.36t', 'time_start': '2017-07-01 00:00:00', 'command_to': 'server', 'location_source': '[d.yangje - d.gwangam][d.yangje - d.gimpo]', 'command': 'calculate_regression', 'time_end': '2017-08-01 00:00:00'}

def getColumnsSource(source):
    result = ''
    list = source.split('[')[1:]
    list = [str.replace(']','') for str in list]
    cnt = 0
    for elt in list:
        cnt = cnt + 1;
        result = result + ', ' + elt + ' AS C' + str(cnt)
    return result

def getColumnsTarget(target):
    print(target)
    result = ''
    result = result + ', ' + target + ' AS T'
    return result

def setWhere(dict):
    result = ''
    return result


def getData(cur, dict):
    source = dict['location_source']
    target = dict['location_target']

    columns_source = getColumnsSource(source)
    columns_target = getColumnsTarget(target)

    print(columns_source)
    print(columns_target)
    print('source:', source)

    query = "\n SELECT D.TIMESTAMP %s" % (columns_source) \
            + "\n  %s" % (columns_target) \
            + "\n  FROM TB_WATER_DISCHARGE D, TB_WATER_PRESSURE P" \
            + "\n  WHERE 1 = 1 " \
            + "\n  AND D.TIMESTAMP = P.TIMESTAMP " \
            + "\n  AND D.TIMESTAMP BETWEEN '%s' AND '%s'" % (comm.getDictValue(dict, 'time_start'), comm.getDictValue(dict, 'time_end')) \
            + comm.getWhereTwoTable('COMMON') \
            + comm.getWhereTwoTable('HONGTONG') \
            + comm.getWhereTwoTable('SEONGSAN_GIMPO')

    print(query)
    cur.execute(query)
    tuple = cur.fetchall()
    df = pd.DataFrame(list(tuple))

    return df

def setA(dict, df):
    source = dict['location_source']
    list = []
    for i in range(len(source.split('[')[1:])):
        list.append('C' + str(i + 1))
    result_df = df.reindex(columns=list)
    result_df['b'] = 1
    return result_df.values

def setY(dict, df):
    source = dict['location_source']
    result_df = df.reindex(columns=['T'])
    return result_df.values

def methodOfLeastSquares(A, Y):
    #A = np.array([[1, 1],[2, 1],[3, 1],[4, 1]])
    #Y = np.array([[1],[2],[3],[4]])
    print(A)
    Apinv = np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)
    x, resid, rank, s = np.linalg.lstsq(A, Y)
    return x


def main():

    con = conn.getConnection()
    cur = con.cursor(MySQLdb.cursors.DictCursor)

    command = 'calculate_regression'
    command_detail = ''

    #['Timestamp','f_hongtong','f_lotte_in','p_oryun_1','p_songpa_1','p_oryun_2','p_songpa_2']

    dict = {}
    dict['command'] = command
    dict['command_to'] = 'server'
    time_start = '2017-07-01 00:00:00'
    time_end = '2017-08-01 00:00:00'
    dict['location_source'] = '[d.P_ORYUN_1 - d.P_SONGPA_1][d.P_ORYUN_2 - d.P_SONGPA_2]'
    dict['location_target'] = 'd.D_HONGTONG + d.D_LOTTE_IN'
    dict['time_start'] = time_start
    dict['time_end'] = time_end

    df = getData(cur, dict)

    print(dict)

    x = methodOfLeastSquares(setA(dict, df), setY(dict, df))
    print(x)

    print(df.head(10))


if __name__ == '__main__':
    main()






