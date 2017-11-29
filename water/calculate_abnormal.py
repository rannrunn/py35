# 모니터링 화면에 사용하는 소스
# 실제 운영서버에서는 사용되지 않지만 남겨놓습니다.
import MySQLdb
import dbconnection as conn
import common as comm
import traceback
import threading


def getCnt(cur, dict):
    result = 0
    query = "\n SELECT COUNT(TIMESTAMP) AS CNT" \
            + "\n FROM TB_WATER_DISCHARGE" \
            + "\n WHERE TIMESTAMP BETWEEN ADDTIME('%s', '-00:04:00') AND '%s'" % (dict['time'], dict['time']) \
            + comm.getWhereOneTable('COMMON') \
            + comm.getWhereOneTable('HONGTONG') \
            + comm.getWhereOneTable('SEONGSAN_GIMPO')
    cur.execute(query)
    if cur.rowcount != 0:
        result = cur.fetchall()[0]['CNT']
    return result

def getLeak(cur, dict, section_one_standard_discharge, section_two_standard_discharge):
    cnt = 0
    query = "\n SELECT " \
            + "\n  " + comm.getSelect('ONE') +  " AS ONE" \
            + "\n  , " + comm.getSelect('TWO') + " AS TWO" \
            + "\n  FROM TB_WATER_DISCHARGE" \
            + "\n  WHERE TIMESTAMP BETWEEN ADDTIME('%s', '-00:05:00') AND '%s'" % (dict['time'], dict['time']) \
            + comm.getWhereOneTable('COMMON') \
            + comm.getWhereOneTable('HONGTONG') \
            + comm.getWhereOneTable('SEONGSAN_GIMPO') \
            + "\n  LIMIT 5;"
    cur.execute(query)
    for row in cur.fetchall():
        if float(row['ONE']) > section_one_standard_discharge:
            print(row['ONE'], ':', section_one_standard_discharge)
            cnt = cnt + 1
        if float(row['TWO']) > section_two_standard_discharge:
            print(row['TWO'], ':', section_two_standard_discharge)
            cnt = cnt + 1
    # 누수가 지점 별로 5번 씩 발생했을 경우 누수 -> result = 1
    result = 1 if cnt == 10 else 0
    return result

def task(cur, dict):
    global t

    # 공급 유량 합계 = 팔당 + 36T
    # SELECT FLOOR(D_PALDANG + D_36T) AS TOTAL_SUPPLY
    # FROM TB_WATER
    # WHERE TIMESTAMP <= '2017-06-01 00:05:00'
    # LIMIT 5

    stddc_hongtong = 0
    stddc_seongsan_gimpo = 0

    section_one_standard_discharge = float(dict['section_one_standard_discharge'])
    section_two_standard_discharge = float(dict['section_two_standard_discharge'])

    cnt_data = getCnt(cur, dict)
    print('cnt_data:', cnt_data)
    if (cnt_data < 5):
        print('err:최근 5분 간의 데이터가 연속적이지 않거나 없는 데이터가 있습니다.')
        return

    #print(query_two)

    cnt_leak = getLeak(cur, dict, section_one_standard_discharge, section_two_standard_discharge)
    print('cnt_leak:', cnt_leak)
    if cnt_leak == 10:
        print('누수가 발생했습니다.')

    t = threading.Timer(60, task, [cur, dict])
    t.start()
    pass


def main():

    con = conn.getConnection()
    cur = con.cursor(MySQLdb.cursors.DictCursor)

    dict = {}
    dict['command'] = 'calculate_abnormal'
    dict['command_to'] = 'server'
    dict['section_one_standard_discharge'] = '1539'
    dict['section_two_standard_discharge'] = '3981'


    # 테스트
    dict['time'] = '2017-06-01 00:06:00'

    try:

        task(cur, dict)

        print('..')
    except Exception as e:
        print('err:Exception')
        traceback.print_exc()
        t.cancel()
    pass

if __name__ == '__main__':
    main()

