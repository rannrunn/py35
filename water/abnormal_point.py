import MySQLdb
import dbconnection as conn
import common as comm
import traceback

def getStatistics():
    dict = {}
    query = "\n SELECT " \
                + "\n  AVG_HONGTONG, STDDEV_HONGTONG, STDDC_HONGTONG, AVG_SEONGSAN_GIMPO, STDDEV_SEONGSAN_GIMPO, STDDC_SEONGSAN_GIMPO " \
                + "\n  FROM TB_WATER_TWO " \
                + "\n  WHERE TIME_START = '%s' AND TIME_END = '%s' " % (time_start, time_end)
    cur.execute(query)
    if cur.rowcount != 0:
        dict = cur.fetchall()[0]
    return dict

def getCnt():
    result = 0
    query = "\n SELECT COUNT(TIMESTAMP) AS CNT" \
            + "\n FROM TB_WATER" \
            + "\n WHERE TIMESTAMP BETWEEN ADDTIME('%s', '-00:04:00') AND '%s'" % (time, time) \
            + comm.getWhere('COMMON') \
            + comm.getWhere('HONGTONG') \
            + comm.getWhere('SEONGSAN_GIMPO')
    cur.execute(query)
    if cur.rowcount != 0:
        result = cur.fetchall()[0]['CNT']
    return result

def getLeak(stddc_hongtong, stddc_seongsan_gimpo):
    result = 0
    query = "\n SELECT " \
            + "\n  " + comm.getSelect('HONGTONG') +  " AS HONGTONG" \
            + "\n  , " + comm.getSelect('SEONGSAN_GIMPO') + " AS SEONGSAN_GIMPO" \
            + "\n  FROM TB_WATER" \
            + "\n  WHERE TIMESTAMP BETWEEN ADDTIME('%s', '-00:05:00') AND '%s'" % (time, time) \
            + comm.getWhere('COMMON') \
            + comm.getWhere('HONGTONG') \
            + comm.getWhere('SEONGSAN_GIMPO') \
            + "\n  LIMIT 5;"
    cur.execute(query)
    for row in cur.fetchall():
        if float(row['HONGTONG']) > stddc_hongtong:
            print(row['HONGTONG'], ':', stddc_hongtong)
            result = result + 1
        if float(row['SEONGSAN_GIMPO']) > stddc_seongsan_gimpo:
            print(row['SEONGSAN_GIMPO'], ':', stddc_seongsan_gimpo)
            result = result + 1
    return result


def main():
    try:
        # 공급 유량 합계 = 팔당 + 36T
        # SELECT FLOOR(D_PALDANG + D_36T) AS TOTAL_SUPPLY
        # FROM TB_WATER
        # WHERE TIMESTAMP <= '2017-06-01 00:05:00'
        # LIMIT 5

        stddc_hongtong = 0
        stddc_seongsan_gimpo = 0

        dict_stats = getStatistics()

        if bool(dict_stats):
            stddc_hongtong = float(dict_stats['STDDC_HONGTONG'])
            stddc_seongsan_gimpo = float(dict_stats['STDDC_SEONGSAN_GIMPO'])
            print(dict_stats)

        cnt_data = getCnt()
        print('cnt_data:', cnt_data)
        if (cnt_data < 5):
            print('err:최근 5분 간의 데이터가 연속적이지 않거나 없는 데이터가 있습니다.')
            return

        #print(query_two)

        cnt_leak = getLeak(stddc_hongtong, stddc_seongsan_gimpo)
        print('cnt_leak:', cnt_leak)
        if cnt_leak == 10:
            print('누수가 발생했습니다.')

        print('..')
    except Exception as e:
        print('err:Exception')
        traceback.print_exc()
    pass

if __name__ == '__main__':

    time = '2017-06-01 00:06:00'
    time_start = '2017-06-01 00:01:00'
    time_end = '2017-08-11 00:00:00'

    con = conn.getConnection()
    cur = con.cursor(MySQLdb.cursors.DictCursor)

    main()

