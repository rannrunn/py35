import MySQLdb
import dbconnection as conn
import common as comm
import traceback

def getData():
    dict = {}
    query = "\n SELECT " \
                + "\n  AVG_HONGTONG, STDDEV_HONGTONG, STDDC_HONGTONG, AVG_SEONGSAN_GIMPO, STDDEV_SEONGSAN_GIMPO, STDDC_SEONGSAN_GIMPO " \
                + "\n  FROM TB_WATER_TWO " \
                + "\n  WHERE TIME_START = '%s' AND TIME_END = '%s' " % (time_start, time_end)
    cur.execute(query)
    if cur.rowcount != 0:
        dict = cur.fetchall()[0]
    return dict

def main():
    try:
        # 공급 유량 합계 = 팔당 + 36T
        # SELECT FLOOR(D_PALDANG + D_36T) AS TOTAL_SUPPLY
        # FROM TB_WATER
        # WHERE TIMESTAMP <= '2017-06-01 00:05:00'
        # LIMIT 5
        query_one = "\n SELECT COUNT(TIMESTAMP) AS CNT" \
                    + "\n FROM TB_WATER" \
                    + "\n WHERE TIMESTAMP BETWEEN ADDTIME('%s', '-00:05:00') AND '%s'" % (time, time) \
                    + comm.getWhere('COMMON') \
                    + comm.getWhere('HONGTONG') \
                    + comm.getWhere('SEONGSAN_GIMPO')

        query_two = "\n SELECT " \
                    + "\n  " + comm.getSelect('HONGTONG') +  " AS HONGTONG" \
                    + "\n  , " + comm.getSelect('SEONGSAN_GIMPO') + " AS SEONGSAN_GIMPO" \
                    + "\n  FROM TB_WATER" \
                    + "\n  WHERE TIMESTAMP BETWEEN ADDTIME('%s', '-00:05:00') AND '%s'" % (time, time) \
                    + comm.getWhere('COMMON') \
                    + comm.getWhere('HONGTONG') \
                    + comm.getWhere('SEONGSAN_GIMPO') \
                    + "\n  LIMIT 5;"

        dict = getData()

        avg_hongtong = 0
        stddev_hongtong = 0
        stddc_hongtong = 0
        avg_seongsan_gimpo = 0
        stddev_seongsan_gimpo = 0
        stddc_seongsan_gimpo = 0

        if bool(dict):
            avg_hongtong = float(dict['AVG_HONGTONG'])
            stddev_hongtong = float(dict['STDDEV_HONGTONG'])
            stddc_hongtong = float(dict['STDDC_HONGTONG'])
            avg_seongsan_gimpo = float(dict['AVG_SEONGSAN_GIMPO'])
            stddev_seongsan_gimpo = float(dict['STDDEV_SEONGSAN_GIMPO'])
            stddc_seongsan_gimpo = float(dict['STDDC_SEONGSAN_GIMPO'])
            print(dict)

        cur.execute(query_one)
        cnt_data = int(cur.fetchall()[0]['CNT'])
        print('cnt_data:', cnt_data)
        if (cnt_data < 5):
            print('err:최근 5분 간의 데이터가 연속적이지 않거나 없는 데이터가 있습니다.')
            return

        #print(query_two)
        cnt_nusu = 0
        cur.execute(query_two)
        for row in cur.fetchall():
            if float(row['HONGTONG']) > stddc_hongtong:
                cnt_nusu = cnt_nusu + 1
            if float(row['SEONGSAN_GIMPO']) > stddc_hongtong:
                cnt_nusu = cnt_nusu + 1

        if cnt_nusu == 5:
            print('누수가 발생했습니다.')

        print('..')
    except Exception as e:
        print('err:Exception')
        traceback.print_exc()
    pass

if __name__ == '__main__':

    time = '2017-06-01 00:05:00'
    time_start = '2017-06-01 00:01:00'
    time_end = '2017-08-11 00:00:00'

    con = conn.getConnection()
    cur = con.cursor(MySQLdb.cursors.DictCursor)

    main()

