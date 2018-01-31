﻿# 회귀 분석을 하는 소스

import MySQLdb
import dbconnection as conn
import numpy as np

from project.water_leak import common as comm


# 회귀 분석을 할 A 값 들의 리스트
def setA(df):
    result_df = df.reindex(columns=list(df.columns[0:-1]))
    result_df['b'] = 1
    return result_df.values

# 회귀 분석을 할 Y 값의 리스트
def setY(df):
    result_df = df.reindex(columns=list(df.columns[-1:]))
    return result_df.values

# 최소제곱법을 이용해 회귀분석
def methodOfLeastSquares(A, Y):
    x = np.array([])
    try:
        x, resid, rank, s = np.linalg.lstsq(A, Y)
    except Exception as e:
        pass
    return x

# 다중상관계수 계산
def getMultipleCorrelation(df, result_regression):
    # 회귀계산의 독립변수로 사용된 데이터 추출
    df_one = df.reindex(columns=list(df.columns[0:-1]))
    df_one['b'] = 1
    # 회귀계산의 종속변수로 사용된 데이터 추출
    df_two = df.reindex(columns=list(df.columns[-1:]))
    # 추출된 값과 회귀계수를 행렬연산하여 데이터 프레임에 추가
    df_two['PREDICTION_DATA'] = np.dot(df_one.values, result_regression)
    A = df_two.iloc[:,0]
    B = df_two.iloc[:,1]
    # 다중상관계수를 계산하여 리턴
    return round(A.corr(B), 4)

# 결정계수 계산
def getRSquare(result_multiple_correlation):
    return round(result_multiple_correlation ** 2, 4)


# 회귀분석에 대한 계산을 한다.
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

        # 회귀 분석에 대한 결과를 가져온다.
        result_regression = methodOfLeastSquares(setA(df), setY(df))
        #print('result_regression', result_regression)

        # 회귀 분석의 결과값이 없을 경우 '데이터를 확인해 보세요'를 반환
        if len(result_regression) == 0:
            dict['return_value'] = 'Check the data'
            return dict

        # 다중 상관계수 계산
        result_multiple_correlation = getMultipleCorrelation(df, result_regression)
        #print('result_multiple_correlation', result_multiple_correlation)

        # 결정 계수 계산 : 결정계수는 다중상관계수의 제곱이다.
        result_r_square = getRSquare(result_multiple_correlation)
        #print('result_r_square', result_r_square)

        # JSON 생성
        return_key_result_regression = ''
        return_value_result_regression = ''
        # weight 와 bias
        for idx in range(len(df.columns) - 1):
            return_key_result_regression += 'weight_' + str(idx + 1) + ','
        return_key_result_regression += 'bias'
        # 회귀계수
        for idx in range(len(result_regression)):
            return_value_result_regression += str(round(result_regression[idx][0], 4)) + ','
        return_value_result_regression = return_value_result_regression[:-1]

        dict['return_value'] = return_key_result_regression + ':' + return_value_result_regression + '\nmultiple_correlation:' + str(result_multiple_correlation) + '\nr_square:' + str(result_r_square)

    except Exception as e:
        import traceback
        traceback.print_exc()
        if dict['error'] == '':
            dict['error'] = 'calculate regression error'
    finally:
        #print(dict)
        con.close()
        return dict

if __name__ == '__main__':
    dict = {}
    dict['command'] = 'calculate_regression'
    dict['command_to'] = 'server'
    dict['sector'] = '1'
    dict['table'] = 'RDR01MI_TB'
    dict['input'] = 'D_PALDANG, D_HONGTONG, D_YANGJECHEON'
    dict['output'] = 'D_PALDANG'
    dict['time_start'] = '2017-06-01 11:22:00'
    dict['time_end'] = '2017-08-27 13:22:00'

    calculate(dict)


