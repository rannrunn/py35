# 회귀 분석을 하는 소스

import MySQLdb
import numpy as np
import pandas as pd

import common as comm
import dbconnection as conn
import traceback

# 회귀 분석을 할 A 값 들의 리스트
def setA(df):
    result_df = df.reindex(columns=list(df.columns[0:-1]))
    result_df['b'] = 1
    return result_df.values

# 회귀 분석을 할 Y 값의 리스트
def setY(df):
    result_df = df.reindex(columns=list(df.columns[-1:]))
    return result_df.values

# pseudo inverse를 이용해 회귀분석
def methodOfLeastSquares(A, Y):
    #A = np.array([[1, 1],[2, 1],[3, 1],[4, 1]])
    #Y = np.array([[1],[2],[3],[4]])
    Apinv = np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)
    x, resid, rank, s = np.linalg.lstsq(A, Y)
    return x

# 다중상관계수 계산
def getMultipleCorrelation(df, result_regression):
    # 회귀계산의 독립변수로 사용된 데이터 추출
    df_one = df.reindex(columns=list(df.columns[0:-1]))
    df_one['b'] = 1
    # 회귀계산의 종속변수로 사용된 데이터 추출
    df_two = df.reindex(columns=list(df.columns[-1:]))
    # 추출된 값과 회귀계수를 행렬연산하여 데이터 프레임에 추가
    df_two['RESULT_CAL'] = np.dot(df_one.values, result_regression)
    A = df_two.iloc[:,0]
    B = df_two.iloc[:,1]
    # 다중상관계수를 계산하여 리턴
    return A.corr(B)

# 결정계수 계산
def getRSquare(result_multiple_correlation):
    return result_multiple_correlation ** 2


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

        # 지점 리스트를 불러온다.
        list_sector = comm.getSector(cur, dict)
        # 데이터를 불러온다.
        df = comm.getDataFrame(cur, dict, list_sector)

        # 회귀 분석에 대한 결과를 가져온다.
        result_regression = methodOfLeastSquares(setA(df), setY(df))
        #print('result_regression', result_regression)

        # 다중 상관계수 계산
        result_multiple_correlation = getMultipleCorrelation(df, result_regression)
        print('result_multiple_correlation', result_multiple_correlation)

        # 결정 계수 계산 : 결정계수는 다중상관계수의 제곱이다.
        result_r_square = getRSquare(result_multiple_correlation)
        print('result_r_square', result_r_square)
        print(df.columns)
        print(result_regression)
        # JSON 생성
        return_key_result_regression = ''
        return_value_result_regression = ''
        # weight 와 bias
        for idx in range(len(df.columns) - 1):
            return_key_result_regression += 'weight_' + str(idx + 1) + ','
        return_key_result_regression += 'bias'
        # 회귀계수
        for idx in range(len(result_regression)):
            return_value_result_regression += str(result_regression[idx][0]) + ','
        return_value_result_regression = return_value_result_regression[:-1]

        dict['return_value'] = {return_key_result_regression:return_value_result_regression,'multiple_correlation':result_multiple_correlation,'r_square':result_r_square}

        print(dict)

        return dict

    except Exception as e:
        traceback.print_exc()

if __name__ == '__main__':
    dict = {}
    dict['command'] = 'calculate_regression'
    dict['command_to'] = 'server'
    dict['sector'] = '1'
    dict['table'] = 'RDR01MI_TB'
    dict['time_start'] = '2017-06-01 11:22:00'
    dict['time_end'] = '2017-08-27 13:22:00'

    calculate(dict)






