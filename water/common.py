import pandas as pd
import datetime

# value 를 가져온다.
def getDictValue(dict, key):
    return dict[key] if key in dict else ''

# 확인요망
# DB 에서 구간에 속한 지점을 가져온다.
def getSector(cur, dict):
    list_col = ['STEP_IN_1', 'STEP_IN_2', 'STEP_IN_3', 'STEP_IN_4', 'STEP_IN_5', 'STEP_OUT_1']
    list_return = []
    query = """
                SELECT STEP_IN_1, STEP_IN_2, STEP_IN_3, STEP_IN_4, STEP_IN_5, STEP_OUT_1  
                FROM SEC_TB
                WHERE 1 = 1
                AND PIPE_BLK = '%s'
            """ % (getDictValue(dict, 'sector'))
    cur.execute(query)
    tuple = cur.fetchall()
    if cur.rowcount == 1:
        for item in list_col:
            if tuple[0][item] != '' and tuple[0][item] is not None:
                list_return.append(tuple[0][item])
    return list_return

# DB에서 지점의 데이터를 가져온다.
def getLocationData(cur, dict, tag_name):
    query = """
                SELECT DATE_FORMAT(LOG_TIME, '%%Y-%%m-%%d %%H:%%i:%%s') AS LOG_TIME, TAG_VAL  
                FROM %s
                WHERE 1 = 1
                AND TAG_NAME = '%s'
                AND LOG_TIME BETWEEN DATE_FORMAT('%s', '%%y%%m%%d%%H%%i%%s') AND DATE_FORMAT('%s', '%%y%%m%%d%%H%%i%%s')                
            """ % (getDictValue(dict, 'table'), tag_name, getDictValue(dict, 'time_start'), getDictValue(dict, 'time_end'))
    cur.execute(query)
    tuple = cur.fetchall()
    return pd.DataFrame(list(tuple))


# 인덱스를 생성하기 위해 시작시간과 종료시간이 몇 분 차이나는 지 구한다.
def getTimeDifference(dict):
    time_start = datetime.datetime.strptime(getDictValue(dict, 'time_start'), '%Y-%m-%d %H:%M:%S')
    time_end = datetime.datetime.strptime(getDictValue(dict, 'time_end'), '%Y-%m-%d %H:%M:%S')
    time_diff = time_end - time_start
    return int(time_diff.total_seconds() / 60) + 1


def getDataFrame(cur, dict, list_sector):
    # 시작시간과 종료시간이 몇 분 차이나는 지 구한다.
    diff = getTimeDifference(dict)
    # 인덱스를 생성한다.
    index_minutes = pd.date_range(getDictValue(dict, 'time_start'), periods=diff, freq='1min')
    # 인덱스를 가지고 데이터 프레임을 생성한다.
    df = pd.DataFrame(index=index_minutes)
    # 지점 리스트를 불러온다.
    list_sector = getSector(cur, dict)
    # 지점 데이터를 데이터 프레임에 추가한다.
    for item in list_sector:
        df_sector = pd.DataFrame()
        # 지점의 데이터를 가져온다.
        df_sector = getLocationData(cur, dict, item)
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
    return df