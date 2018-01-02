import time

import MySQLdb
import pandas as pd

# SELECT
def selectPoleSensor(cur):
    query = """SELECT POLE_ID, SENSOR_ID
                FROM TB_IOT_POLE_SECOND_POLE_SENSOR
                WHERE POLE_ID <> ''
                AND POLE_ID NOT LIKE '0000%'
                AND POLE_ID NOT LIKE 'TEST%'
                AND POLE_ID NOT LIKE 'DEMO%'
                """
    cur.execute(query);
    results = cur.fetchall()
    df = pd.DataFrame(list(results))
    return df


# SELECT
def selectData(cur, pole_id, sensor_id, time_start, time_end):
    query = """SELECT *
                FROM TB_IOT_POLE_SECOND
                WHERE POLE_ID = '%s'
                AND SENSOR_ID = '%s'
                AND TIME_ID BETWEEN '%s' AND '%s'
                """ % (pole_id, sensor_id, time_start, time_end )
    cur.execute(query);
    results = cur.fetchall()
    df = pd.DataFrame(list(results))
    return df

def getData(cur, pole_id, sensor_id, time_start, time_end, resample_how):
    df_sel = selectData(cur, pole_id, sensor_id, time_start, time_end)

    # 1. 얕은복사를 해야 경고 메시지가 뜨지 않는다.
    # 2. 얕은복사를 하면 데이터 처리 속도가 빨라진다.
    df_pole = df_sel[['TIME_ID','TEMP']].copy()
    # inplace True => 제자리에서 변경(새로운 DataFrame을 생성하지 않음)
    df_pole.set_index('TIME_ID', inplace=True)
    df_pole.index = pd.to_datetime(df_pole.index)

    # (df_pole['TEMP'] != '') => None도 이 조건에 해당되기 때문에 잘 구분해서 써야 한다.
    df_pole.loc[:, 'TEMP'] = 1
    # NaN은 float 형식이기 때문에 NaN이 있을 경우 integer로는 변환할 수 없다.
    df_pole = df_pole.astype(float)

    # resample을 하기 위해서는 인덱스의 형식이 datetime 형식이어야 한다.
    df_pole = df_pole.resample(resample_how).max()
    df_pole.columns = [pole_id + '_' + sensor_id]

    return df_pole


if __name__ == '__main__':

    start_time_main = time.time()

    # DB connection
    con = MySQLdb.connect('localhost', 'root', '1111', 'kepco')
    con.set_character_set('utf8')
    cur = con.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SET NAMES utf8;')
    cur.execute('SET CHARACTER SET utf8;')
    cur.execute('SET character_set_connection=utf8;')

    time_start = '2017-06-19 00:00:00'
    time_end = '2017-12-27 23:59:59'
    resample_how = '10T'

    path_dir = 'G:\\'
    file_name = '폴의센서별10분당데이터유무.csv'

    try:
        df_pole_sensor = selectPoleSensor(cur)
        df_left = pd.DataFrame()
        df_right = pd.DataFrame()

        for idx in range(len(df_pole_sensor)):
            pole_id = df_pole_sensor['POLE_ID'][idx]
            sensor_id = df_pole_sensor['SENSOR_ID'][idx]
            df_right = getData(cur, pole_id, sensor_id, time_start, time_end, resample_how)
            df_left = pd.merge(df_left, df_right, left_index=True, right_index=True, how='outer').copy()
            cnt = idx + 1
            print('Count:' + str(cnt) +' / Past Time:' + str(round(time.time() - start_time_main, 4)))

        df_left.to_csv(path_dir + file_name)

        print('Total Time:' + str(round(time.time() - start_time_main, 4)))

        con.close()
    except Exception as e:
        print('Exception')
    finally:
        con.close()

