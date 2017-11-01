import MySQLdb
import numpy as np
import pandas as pd
from numpy import*

import common as comm
import dbconnection as conn

A = np.array([[2, 0], [-1, 1], [0, 2]])
print(A)
b = np.array([[1], [0], [-1]])
Apinv = np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)
print(Apinv)
x, resid, rank, s = np.linalg.lstsq(A, b)
print(x)

# {'location_target': 'd.paldang + d.36t', 'time_start': '2017-07-01 00:00:00', 'command_to': 'server', 'location_source': '[d.yangje - d.gwangam][d.yangje - d.gimpo]', 'command': 'calculate_regression', 'time_end': '2017-08-01 00:00:00'}

def methodOfLeastSquares():
    A = np.array([[1, 1],[2, 1],[3, 1],[4, 1]])
    b = np.array([[1],[2],[3],[4]])
    Apinv = np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)
    x, resid, rank, s = np.linalg.lstsq(A, b)
    return x

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

    #len = len(dict['location_source'])
    #print('xx:',len)

    result = ''

    return result
    pass

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

    #print(query)
    cur.execute(query)
    tuple = cur.fetchall()
    df = pd.DataFrame(list(tuple))

    return df


def main():

    con = conn.getConnection()
    cur = con.cursor(MySQLdb.cursors.DictCursor)

    command = 'calculate_regression'
    command_detail = ''

    dict = {}
    dict['command'] = command
    dict['command_to'] = 'server'
    time_start = '2017-07-01 00:00:00'
    time_end = '2017-08-01 00:00:00'
    dict['location_source'] = '[d.D_PALDANG - d.D_36T][d.D_GWANGAM - d.D_YANGJE_23T]'
    dict['location_target'] = 'd.D_PALDANG + d.D_36T'
    dict['time_start'] = time_start
    dict['time_end'] = time_end


    df = getData(cur, dict)
    setWhere(dict)

    print(dict)
    print('xx', df)

    methodOfLeastSquares()


if __name__ == '__main__':
    main()






