# coding: utf-8
import time

import MySQLdb
import pandas as pd

# SELECT
def select(cur, pole_id, time_start, time_end):
    query = """SELECT *
                FROM TB_IOT_POLE_total_201604
                WHERE POLE_ID = '%s'  
                and part_name = '변압기 본체'
                """ % (pole_id)
    cur.execute(query);
    results = cur.fetchall()
    df_query = pd.DataFrame(list(results))
    return df_query

if __name__ == '__main__':

    start = time.time()

    # DB connection
    con = MySQLdb.connect('localhost', 'root', '1111', 'kepco')
    con.set_character_set('utf8')
    cur = con.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SET NAMES utf8;')
    cur.execute('SET CHARACTER SET utf8;')
    cur.execute('SET character_set_connection=utf8;')

    # 변수 설정
    pole_id = '8132X291'
    time_date = '20161112'
    time_start = time_date + '0000'
    time_end = time_date + '2359'

    try:
        # 쿼리 SELECT
        df_query = select(cur, pole_id, time_start, time_end)
        print('csv로 만듭니다.')
        df_query.to_csv('c:\\tmp\\8132X291.csv')
        con.close()
    except Exception as e:
        print('Exception')
    finally:
        con.close()



