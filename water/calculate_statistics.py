# 통계 계산을 하는 소스
# 확인 요망을 주석으로 단 부분은 서비스 전에 확인해야 한다.

import MySQLdb
import pandas as pd

import common as comm
import dbconnection as conn
import traceback


# 상관계수 : 지점 간의 상관계수를 모두 구해야 한다.
def getCorrelation(cur, dict, df):
    list_sector = comm.getSector(cur, dict)
    dict_return_value = {}
    for idx_one in range(len(list_sector)):
        for idx_two in range(idx_one + 1, len(list_sector)):
            dict_return_value[list_sector[idx_one] + ',' + list_sector[idx_two]] = df[list_sector[idx_one]].corr(df[list_sector[idx_two]])
    dict['return_value'] = {'correlation':dict_return_value}
    return dict

# 요청에 따라 계산을 수행한다.
def calculate(dict):

    try:

        # DB 커넥션 을 생성한다.
        con = conn.getConnection()
        con.set_character_set('utf8')
        cur = con.cursor(MySQLdb.cursors.DictCursor)
        cur.execute('SET NAMES utf8;')
        cur.execute('SET CHARACTER SET utf8;')
        cur.execute('SET character_set_connection=utf8;')

        command_detail = comm.getDictValue(dict, 'command_detail')

        dict['command_to'] = 'client'

        # 지점 리스트를 불러온다.
        list_sector = comm.getSector(cur, dict)
        # 데이터를 불러온다.
        df = comm.getDataFrame(cur, dict, list_sector)

        # 평균
        if command_detail == 'average':
            list_sector = comm.getSector(cur, dict)
            dict_return_value = {}
            for item in list_sector:
                series = df[item]
                dict_return_value[item] = series.mean()
            dict['return_value'] = {'average':dict_return_value}
        # 분산
        elif command_detail == 'variance':
            list_sector = comm.getSector(cur, dict)
            dict_return_value = {}
            for item in list_sector:
                series = df[item]
                dict_return_value[item] = series.var()
            dict['return_value'] = {'variance':dict_return_value}
        # 표준편차
        elif command_detail == 'standard_deviation':
            list_sector = comm.getSector(cur, dict)
            dict_return_value = {}
            for item in list_sector:
                series = df[item]
                dict_return_value[item] = series.std()
            dict['return_value'] = {'standard_deviation':dict_return_value}
        # 상관계수
        elif command_detail == 'correlation':
            dict = getCorrelation(cur, dict, df)

        print(dict)

        return dict

    except Exception as e:
        traceback.print_exc()

if __name__ == '__main__':
    dict = {}
    dict['command'] = 'calculate_statistics'
    dict['command_to'] = 'server'
    dict['command_detail'] = 'correlation'
    dict['sector'] = '1'
    dict['table'] = 'RDR01MI_TB'
    dict['time_start'] = '2017-06-01 11:22:00'
    dict['time_end'] = '2017-08-27 13:22:00'

    calculate(dict)