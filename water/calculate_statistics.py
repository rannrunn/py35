﻿# 통계 계산을 하는 소스

import traceback

import MySQLdb

import common as comm
import dbconnection as conn


# 상관계수 : 지점 간의 상관계수를 모두 구해야 한다.
def getCorrelation(cur, dict, df):
    dict_return_value = ''
    for idx_one in range(len(df.columns)):
        for idx_two in range(idx_one + 1, len(df.columns)):
            dict_return_value += '\'' + df.columns[idx_one] + ',' + df.columns[idx_two] + '\':\'' + str(df[df.columns[idx_one]].corr(df[df.columns[idx_two]])) + '\','
    # 쉼표 제거
    dict_return_value = dict_return_value[:-1]
    dict['return_value'] = 'correlation:{' + dict_return_value + '}'
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

        # 데이터를 불러온다.
        df = comm.getDataFrame(cur, dict)

        if 'error' in dict and dict['error'] != '':
            raise Exception

        # 평균
        if command_detail == 'average':
            dict_return_value = ''
            for item in df.columns:
                series = df[item]
                dict_return_value += '\'' + item + '\':\'' + str(series.mean()) + '\','
            # 쉼표 제거
            dict_return_value = dict_return_value[:-1]
            dict['return_value'] = 'average:{' + dict_return_value + '}'
        # 분산
        elif command_detail == 'variance':
            dict_return_value = ''
            for item in df.columns:
                series = df[item]
                dict_return_value += '\'' + item + '\':\'' + str(series.var()) + '\','
            # 쉼표 제거
            dict_return_value = dict_return_value[:-1]
            dict['return_value'] = 'variance:{' + dict_return_value + '}'
        # 표준편차
        elif command_detail == 'standard_deviation':
            dict_return_value = ''
            for item in df.columns:
                series = df[item]
                dict_return_value += '\'' + item + '\':\'' + str(series.std()) + '\','
            # 쉼표 제거
            dict_return_value = dict_return_value[:-1]
            dict['return_value'] = 'standard_deviation:{' + dict_return_value + '}'
        # 상관계수
        elif command_detail == 'correlation':
            dict = getCorrelation(cur, dict, df)

    except Exception as e:
        #traceback.print_exc()
        if dict['error'] == '':
            dict['error'] = 'calculate statistics error'
    finally:
        #print(dict)
        return dict

if __name__ == '__main__':
    dict = {}
    dict['command'] = 'calculate_statistics'
    dict['command_to'] = 'server'
    # average, variance, standard_deviation, correlation
    dict['command_detail'] = 'standard_deviation'
    dict['sector'] = '1'
    dict['table'] = 'RDR01MI_TB'
    dict['input'] = 'D_GWANGAM, D_HONGTONG, D_YANGJECHEON'
    dict['output'] = 'D_PALDANG'
    dict['time_start'] = '2017-06-01 11:22:00'
    dict['time_end'] = '2017-08-27 13:22:00'

    calculate(dict)
