# 모니터링 화면에 사용하는 소스
# 실제 운영서버에서는 사용되지 않지만 남겨놓습니다.
# coding: utf-8

import MySQLdb
import dbconnection as conn
import traceback
import common as comm

# 평균과 표준편차를 구하는 함수
def getAvgStddev(cur, dict, section, func):
    query = ''
    # HONGTONG = 팔당 + 36T - (광암 + 홍통1 + 홍통2 + 홍통3 + 양재천)
    # 개발 DB 는 홍통유량이 합계되어 저장되어 있는 걸 가정하여 개발하였기 때문에 후에 홍통1 + 홍통2 + 홍통3 유량으로 수정해야 함
    # SEONGSAN = 팔당 + 36T - (광암 + 성산 + 김포 + 양재천)
    # 개발 DB 는 성산김포유량이 합계되어 저장되어 있는 걸 가정하여 개발하였기 때문에 후에 성산 + 김포 유량으로 수정해야 함
    # REGEX : ^[0-9]+\\.?[0-9]*$ : 숫자로 시작하고 마침표가 하나 있거나 없으며 이후 숫자가 0개 이상 오는 경우

    query = "\nSELECT %s(%s) AS %s" % (func, comm.getSelect(section), func) \
            + "\n  FROM TB_WATER_DISCHARGE" \
            + "\n  WHERE 1 = 1" \
            + "\n    AND TIMESTAMP BETWEEN '%s' AND '%s'" % (dict['time_start'], dict['time_end']) \
            + comm.getWhereOneTable('COMMON') \
            + comm.getWhereOneTable('HONGTONG') \
            + comm.getWhereOneTable('SEONGSAN_GIMPO')

    result = 0
    cur.execute(query)
    tuple = cur.fetchall()
    if cur.rowcount != 0 and tuple[0] is not None:
        result = int(tuple[0][func])

    return result

def getStatistics(cur, dict, section):

    # 평균 :
    func = 'AVG'
    avg = getAvgStddev(cur, dict, section, func)

    # 표준편차 :
    func = 'STDDEV'
    stddev = getAvgStddev(cur, dict, section, func)

    # 기준유량 = 평균 + 표준편차
    stddc = avg + stddev

    dict['return_value'] = {}
    dict['return_value']['average'] = avg
    dict['return_value']['standard_deviation'] = stddev
    dict['return_value']['standard_discharge'] = stddc

    return dict

def main():

    con = conn.getConnection()
    con.set_character_set('utf8')
    cur = con.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SET NAMES utf8;')
    cur.execute('SET CHARACTER SET utf8;')
    cur.execute('SET character_set_connection=utf8;')

    dict = {}
    dict['command'] = 'calculate_statistics_abnormal'
    dict['command_to'] = 'server'
    dict['section'] = 'one'
    dict['time_start'] = '2017-06-01 00:01:00'
    dict['time_end'] = '2017-08-11 00:00:00'

    try:

        if dict['section'] == 'one':
            dict = getStatistics(cur, dict, 'ONE')
        elif dict['section'] == 'two':
            dict = getStatistics(cur, dict, 'TWO')

        print(dict)

    except Exception as e:
        print('Ex : avg_stddev_stddc')
        traceback.print_exc()


if __name__ == '__main__':

    main()


