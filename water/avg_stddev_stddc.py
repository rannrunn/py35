# coding: utf-8
import MySQLdb
import dbconnection as conn
import traceback
import common as comm

def getQueryCNT(ratio):
    query =  "\nSELECT CONVERT((COUNT(D_PALDANG) * %s/100) - 1, INT) AS CNT" % (ratio) \
            + "\n  FROM TB_WATER" \
            + "\n  WHERE 1 = 1" \
            + "\n    AND TIMESTAMP BETWEEN '%s' AND '%s'" % (time_start, time_end) \
            + comm.getWhere('COMMON') \
            + comm.getWhere('HONGTONG') \
            + comm.getWhere('SEONGSAN_GIMPO')
    return query

def getQueryQuartile(location, QuartileCNT):
    query =  "\nSELECT" \
            + "\n  DISCHARGE" \
            + "\n  FROM (" \
            + "\n    SELECT" \
            + "\n      %s AS DISCHARGE" % (comm.getSelect(location)) \
            + "\n    FROM TB_WATER" \
            + "\n    WHERE 1 = 1" \
            + "\n    AND TIMESTAMP BETWEEN '%s' AND '%s'" % (time_start, time_end) \
            + comm.getWhere('COMMON') \
            + comm.getWhere('HONGTONG') \
            + comm.getWhere('SEONGSAN_GIMPO') \
            + "\n    ) S1" \
            + "\n  ORDER BY DISCHARGE ASC" \
            + "\n  LIMIT %s, 1;" % (QuartileCNT)
    return query

def getQueryAvgStddev(location, func, quartile1, quartile3):
    query = ''
    # HONGTONG = 팔당 + 36T - (광암 + 홍통1 + 홍통2 + 홍통3 + 양재천)
    # 개발 DB 는 홍통유량이 합계되어 저장되어 있는 걸 가정하여 개발하였기 때문에 후에 홍통1 + 홍통2 + 홍통3 유량으로 수정해야 함
    # SEONGSAN = 팔당 + 36T - (광암 + 성산 + 김포 + 양재천)
    # 개발 DB 는 성산김포유량이 합계되어 저장되어 있는 걸 가정하여 개발하였기 때문에 후에 성산 + 김포 유량으로 수정해야 함
    # REGEX : ^[0-9]+\\.?[0-9]*$ : 숫자로 시작하고 마침표가 하나 있거나 없으며 이후 숫자가 0개 이상 오는 경우
    query = "\nSELECT %s(%s) AS %s" % (func, comm.getSelect(location),func) \
            + "\n  FROM TB_WATER" \
            + "\n  WHERE 1 = 1" \
            + "\n    AND TIMESTAMP BETWEEN '%s' AND '%s'" % (time_start, time_end) \
            + comm.getWhere('COMMON') \
            + comm.getWhere('HONGTONG') \
            + comm.getWhere('SEONGSAN_GIMPO') \
            + "\n    AND %s >= %s" % (comm.getSelect(location), quartile1) \
            + "\n    AND %s <= %s" % (comm.getSelect(location), quartile3)
    return query

def getQuartileCNT(quar):
    result = 0
    cur.execute(getQueryCNT(quar * 25))
    tuple = cur.fetchall()
    if cur.rowcount != 0 and tuple[0]['CNT'] is not None:
        result = int(tuple[0]['CNT'])
    print(quar, ' 사분위 CNT:', result)
    return result

def getQuartile(location, cnt_quartile):
    result = 0
    cur.execute(getQueryQuartile(location, cnt_quartile))
    tuple = cur.fetchall()
    if cur.rowcount != 0 and tuple[0]['DISCHARGE'] is not None:
        result = int(tuple[0]['DISCHARGE'])
    print('유량:', result)
    return result

def getRatioDischarge(location, part):

    cnt_quartile = 0
    partRatio = 0
    result = 0

    if part == 1:
        partRatio = ratio
    elif part == 3:
        partRatio = 100 - ratio

    cur.execute(getQueryCNT(partRatio))
    tuple = cur.fetchall()
    if cur.rowcount != 0 and tuple[0]['CNT'] is not None:
        cnt_quartile = int(tuple[0]['CNT'])
    print(partRatio, ' 퍼센트 CNT:', cnt_quartile)

    cur.execute(getQueryQuartile(location, cnt_quartile))
    tuple = cur.fetchall()
    if cur.rowcount != 0 and tuple[0]['DISCHARGE'] is not None:
        result = int(tuple[0]['DISCHARGE'])
    print(partRatio, ' 퍼센트 유량:', result)
    return result

# 평균과 표준편차를 구하는 함수
def getAvgStddev(location, func, quartile1, quartile3):

    result = 0

    cur.execute(getQueryAvgStddev(location, func, quartile1, quartile3))
    tuple = cur.fetchall()
    if cur.rowcount != 0 and tuple[0][func] is not None:
        result = int(tuple[0][func])

    return result

def mergeData(dict):
    query = "SELECT TIME_START, TIME_END FROM TB_WATER_TWO WHERE TIME_START = '%s' AND TIME_END = '%s'" % (time_start, time_end)
    cur.execute(query);
    tuple = cur.fetchall()
    if cur.rowcount == 0:
        query = "INSERT INTO TB_WATER_TWO(TIME_START, TIME_END, AVG_HONGTONG, STDDEV_HONGTONG, STDDC_HONGTONG, AVG_SEONGSAN_GIMPO, STDDEV_SEONGSAN_GIMPO, STDDC_SEONGSAN_GIMPO) " \
                + "VALUES ('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')" \
                % (time_start, time_end, dict['AVG_HONGTONG'], dict['STDDEV_HONGTONG'], dict['STDDC_HONGTONG'], dict['AVG_SEONGSAN_GIMPO'], dict['STDDEV_SEONGSAN_GIMPO'], dict['STDDC_SEONGSAN_GIMPO'])
        cur.execute(query)
        con.commit()
    else:
        # 업데이트 컬럼이 중복되면 에러가 발생하는 것이 아니라 엉뚱한 컬럼을 업데이트 합니다. (ex:TIME_START가 현재 시간으로 업데이트 됨)
        query = "UPDATE TB_WATER_TWO SET AVG_HONGTONG = '%s', STDDEV_HONGTONG = '%s', STDDC_HONGTONG = '%s', AVG_SEONGSAN_GIMPO = '%s', STDDEV_SEONGSAN_GIMPO = '%s', STDDC_SEONGSAN_GIMPO = '%s'" \
                % (dict['AVG_HONGTONG'], dict['STDDEV_HONGTONG'], dict['STDDC_HONGTONG'], dict['AVG_SEONGSAN_GIMPO'], dict['STDDEV_SEONGSAN_GIMPO'], dict['STDDC_SEONGSAN_GIMPO']) \
                + " WHERE TIME_START = '%s' AND TIME_END = '%s'" % (time_start, time_end)
        cur.execute(query)
        con.commit()

def setData(location):

    dict = {}

    # 제1사분위수
    quartile1 = getQuartile(location, getQuartileCNT(1))
    # 제3사분위수
    quartile3 = getQuartile(location, getQuartileCNT(3))

    # 머리 유량
    headDischarge = getRatioDischarge(location, 1)
    # 꼬리 유량
    tailDischarge = getRatioDischarge(location, 3)

    # 평균 : 분위수를 이용하여 구한다.
    func = 'AVG'
    avg = getAvgStddev(location, func, quartile1, quartile3)

    # 표준편차 : 머리와 꼬리를 일정 비율로 잘라내고 구한다.
    func = 'STDDEV'
    stddev = getAvgStddev(location, func, headDischarge, tailDischarge)

    # 기준유량 = 평균 + 표준편차
    stddc = avg + stddev

    dict['AVG_' + location.upper()] = avg
    dict['STDDEV_' + location.upper()] = stddev
    dict['STDDC_' + location.upper()] = stddc

    print('LOCATION:', location)
    print('AVG:', avg)
    print('STDDEV:', stddev)
    print('STDDC:', stddc)

    return dict

def main():
    try:
        print('LOCATION:','HONGTONG')
        dict_hongttong = setData('HONGTONG')
        print('LOCATION:','SEONGSAN_GIMPO')
        dict_seongsan_gimpo = setData('SEONGSAN_GIMPO')
        dict = {}
        dict.update(dict_hongttong)
        dict.update(dict_seongsan_gimpo)
        mergeData(dict)
    except Exception as e:
        print('Ex : avg_stddev_stddc')
        traceback.print_exc()


if __name__ == '__main__':
    time_start = '2017-06-01 00:01:00'
    time_end = '2017-08-11 00:00:00'
    ratio = 10

    con = conn.getConnection()
    con.set_character_set('utf8')
    cur = con.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SET NAMES utf8;')
    cur.execute('SET CHARACTER SET utf8;')
    cur.execute('SET character_set_connection=utf8;')

    main()

