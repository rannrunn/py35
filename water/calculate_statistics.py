# 통계 계산을 하는 소스
# 확인 요망을 주석으로 단 부분은 서비스 전에 확인해야 한다.

import MySQLdb
import pandas as pd

import common as comm
import dbconnection as conn

import datetime

# 상관관계 : 지점 간의 상관관계를 모두 구해야 한다.
def getCorrelation(cur, dict, df):
    list_sector = comm.getSector(cur, dict)
    dict_return_value = {}
    for idx_one in range(len(list_sector)):
        for idx_two in range(idx_one + 1, len(list_sector)):
            dict_return_value[list_sector[idx_one] + ',' + list_sector[idx_two]] = df[list_sector[idx_one]].corr(df[list_sector[idx_two]])
    dict['return_value'] = {'correlation':dict_return_value}
    return dict

# 인덱스를 생성하기 위해 시작시간과 종료시간이 몇 분 차이나는 지 구한다.
def getTimeDifference(dict):
    time_start = datetime.datetime.strptime(comm.getDictValue(dict, 'time_start'), '%Y-%m-%d %H:%M:%S')
    time_end = datetime.datetime.strptime(comm.getDictValue(dict, 'time_end'), '%Y-%m-%d %H:%M:%S')
    time_diff = time_end - time_start
    return int(time_diff.total_seconds() / 60) + 1

# 요청에 따라 계산을 수행한다.
def calculate(dict):

    # DB 커넥션 을 생성한다.
    con = conn.getConnection()
    con.set_character_set('utf8')
    cur = con.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SET NAMES utf8;')
    cur.execute('SET CHARACTER SET utf8;')
    cur.execute('SET character_set_connection=utf8;')
    
    command_detail = comm.getDictValue(dict, 'command_detail')

    # 시작시간과 종료시간이 몇 분 차이나는 지 구한다.
    diff = getTimeDifference(dict)
    # 인덱스를 생성한다.
    index_minutes = pd.date_range(comm.getDictValue(dict, 'time_start'), periods=diff, freq='1min')
    # 인덱스를 가지고 데이터 프레임을 생성한다.
    df = pd.DataFrame(index=index_minutes)
    # 지점 리스트를 불러온다.
    list_sector = comm.getSector(cur, dict)
    # 지점 데이터를 데이터 프레임에 추가한다.
    cnt = 0
    for item in list_sector:
        df_sector = pd.DataFrame()
        # 지점의 데이터를 가져온다.
        df_sector = comm.getLocationData(cur, dict, item)
        # 인덱스를 LOG_TIME으로 설정한다.
        df_sector.set_index('LOG_TIME', inplace=True)
        # 컬럼명을 바꾼다.
        df_sector.rename(columns={'TAG_VAL': item}, inplace=True)
        # merge를 이용해 데이터를 inner join 한다.
        df = pd.merge(df, df_sector, left_index=True, right_index=True)

    # 데이터 프레임의 데이터 중에 빈 값이 있는 row 는 제거해야 한다.
    for idx in range(len(list_sector)):
        df = df[(df[list_sector[idx]] != '')]
        df = df[df[list_sector[idx]].notnull()]
    # 빈 값을 제거했으므로 데이터 형식을 float64로 바꿔준다.
    df = df.astype('float64')

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
    # 상관관계
    elif command_detail == 'correlation':
        dict = getCorrelation(cur, dict, df)

    print(dict)

    return dict

if __name__ == '__main__':
    dict = {}
    dict['command_to'] = 'server'
    dict['command_detail'] = 'average'
    dict['sector'] = '1'
    dict['table'] = 'RDR01MI_TB'
    dict['time_start'] = '2017-06-01 11:22:00'
    dict['time_end'] = '2017-08-27 13:22:00'

    calculate(dict)