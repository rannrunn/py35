import MySQLdb
import numpy as np
import pandas as pd

import common as comm
import dbconnection as conn
import function.statistics as st


def getAvgVarianceStddev(cur, dict, func, columns):
    # HONGTONG = 팔당 + 36T - (광암 + 홍통1 + 홍통2 + 홍통3 + 양재천)
    # 개발 DB 는 홍통유량이 합계되어 저장되어 있는 걸 가정하여 개발하였기 때문에 후에 홍통1 + 홍통2 + 홍통3 유량으로 수정해야 함
    # SEONGSAN = 팔당 + 36T - (광암 + 성산 + 김포 + 양재천)
    # 개발 DB 는 성산김포유량이 합계되어 저장되어 있는 걸 가정하여 개발하였기 때문에 후에 성산 + 김포 유량으로 수정해야 함
    # REGEX : ^[0-9]+\\.?[0-9]*$ : 숫자로 시작하고 마침표가 하나 있거나 없으며 이후 숫자가 0개 이상 오는 경우
    query = "\nSELECT %s(%s) AS %s" % (func, columns,func) \
            + "\n  FROM TB_WATER" \
            + "\n  WHERE 1 = 1" \
            + "\n    AND TIMESTAMP BETWEEN '%s' AND '%s'" % (comm.getDictValue(dict, 'time_start'), comm.getDictValue(dict, 'time_end')) \
            + comm.getWhere('COMMON') \
            + comm.getWhere('HONGTONG') \
            + comm.getWhere('SEONGSAN_GIMPO')

    result = 0
    cur.execute(query)
    tuple = cur.fetchall()
    if cur.rowcount != 0 and tuple[0][func] is not None:
        result = int(tuple[0][func])
    return result

def getCorrLocation(dict):
    corr_x = ''
    corr_y = ''

    if dict['location_one']['type'] == 'discharge':
        corr_x = 'd.' + dict['location_one']['location']
    elif dict['location_one']['type'] == 'pressure':
        corr_x = 'p.' + dict['location_one']['location']

    if dict['location_two']['type'] == 'discharge':
        corr_y = 'd.' + dict['location_two']['location']
    elif dict['location_two']['type'] == 'pressure':
        corr_y = 'p.' + dict['location_two']['location']

    return corr_x, corr_y

def getCorrelationDataFrame(cur, dict, corr_x, corr_y):
    # HONGTONG = 팔당 + 36T - (광암 + 홍통1 + 홍통2 + 홍통3 + 양재천)
    # 개발 DB 는 홍통유량이 합계되어 저장되어 있는 걸 가정하여 개발하였기 때문에 후에 홍통1 + 홍통2 + 홍통3 유량으로 수정해야 함
    # SEONGSAN = 팔당 + 36T - (광암 + 성산 + 김포 + 양재천)
    # 개발 DB 는 성산김포유량이 합계되어 저장되어 있는 걸 가정하여 개발하였기 때문에 후에 성산 + 김포 유량으로 수정해야 함
    # REGEX : ^[0-9]+\\.?[0-9]*$ : 숫자로 시작하고 마침표가 하나 있거나 없으며 이후 숫자가 0개 이상 오는 경우
    query = "\nSELECT %s, %s" % (corr_x, corr_y) \
            + "\n  FROM TB_WATER_DISCHARGE D, TB_WATER_PRESSURE P " \
            + "\n  WHERE 1 = 1 " \
            + "\n    AND D.TIMESTAMP = P.TIMESTAMP " \
            + "\n    AND D.TIMESTAMP BETWEEN '%s' AND '%s'" % (comm.getDictValue(dict, 'time_start'), comm.getDictValue(dict, 'time_end')) \
            + comm.getWhereTwoTable('COMMON') \
            + comm.getWhereTwoTable('HONGTONG') \
            + comm.getWhereTwoTable('SEONGSAN_GIMPO')

    cur.execute(query)
    tuple = cur.fetchall()
    df = pd.DataFrame(list(tuple))

    return df


def getAverage(cur, dict, columns):
    func = 'AVG'
    result = getAvgVarianceStddev(cur, dict, columns, func)
    return result


def getVariance(cur, dict, columns):
    func = 'VARIANCE'
    result = getAvgVarianceStddev(cur, dict, columns, func)
    return result


def getStandardDeviation(cur, dict, columns):
    func = 'STDDEV'
    result = getAvgVarianceStddev(cur, dict, columns, func)
    return result


def getCorrelation(cur, dict):

    corr_x, corr_y = getCorrLocation(dict)
    df = getCorrelationDataFrame(cur, dict, corr_x, corr_y)

    df_x = df.loc[:,corr_x.split('.')[1]]
    df_x = df_x.astype('float64')
    df_y = df.loc[:,corr_y.split('.')[1]]
    df_y = df_y.astype('float64')

    print(st.correlation(np.array(df_x), np.array(df_y)))


if __name__ == '__main__':
    dict = {}
    dict['command_to'] = 'server'
    time_start = '2017-07-01 00:00:00'
    time_end = '2017-08-01 00:00:00'
    dict['command_detail'] = 'correlation'
    dict['location_one'] = {'location':'P_SINCHEON_1', 'type':'pressure'}
    dict['location_two'] = {'location':'P_SINCHEON_2', 'type':'pressure'}
    dict['time_start'] = time_start
    dict['time_end'] = time_end

    con = conn.getConnection()
    con.set_character_set('utf8')
    cur = con.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SET NAMES utf8;')
    cur.execute('SET CHARACTER SET utf8;')
    cur.execute('SET character_set_connection=utf8;')

    getCorrelation(cur, dict)