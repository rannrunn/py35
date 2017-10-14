# coding: utf-8
import MySQLdb
import dbconnection as conn
import traceback

def getQuery(location, func):
    query = ''
    # HONGTONG = 팔당 + 36T - (광암 + 홍통1 + 홍통2 + 홍통3 + 양재천)
    # 개발 DB 는 홍통유량이 합계되어 저장되어 있는 걸 가정하여 개발하였기 때문에 후에 홍통1 + 홍통2 + 홍통3 유량으로 수정해야 함
    # SEONGSAN = 팔당 + 36T - (광암 + 성산 + 김포 + 양재천)
    # 개발 DB 는 성산김포유량이 합계되어 저장되어 있는 걸 가정하여 개발하였기 때문에 후에 성산 + 김포 유량으로 수정해야 함
    # REGEX : ^[0-9]+\\.?[0-9]*$ : 숫자로 시작하고 마침표가 하나 있거나 없으며 이후 숫자가 0개 이상 오는 경우
    if location == 'hongtong':
        query = "\nSELECT %s(D_PALDANG + D_36T - (D_GWANGAM + D_HONGTONG + D_YANGJECHEON)) AS AVG" % (func) \
                + "\n  FROM TB_WATER" \
                + "\n  WHERE 1 = 1" \
                + "\n  -- 숫자인 데이터만 가져오기" \
                + "\n  AND D_PALDANG REGEXP '^[0-9]+\\.?[0-9]*$'" \
                + "\n  AND D_36T REGEXP '^[0-9]+\\.?[0-9]*$'" \
                + "\n  AND D_GWANGAM REGEXP '^[0-9]+\\.?[0-9]*$'" \
                + "\n  AND D_HONGTONG REGEXP '^[0-9]+\\.?[0-9]*$'" \
                + "\n  AND D_YANGJECHEON REGEXP '^[0-9]+\\.?[0-9]*$'" \
                + "\n  AND TIMESTAMP BETWEEN '%s' AND '%s'" % (time_start, time_end)
    elif location == 'seongsan_gimpo':
        query = "\nSELECT %s(D_PALDANG + D_36T - (D_GWANGAM + D_SEONGSAN_GIMPO + D_YANGJECHEON)) AS AVG" % (func) \
                + "\n  FROM TB_WATER" \
                + "\n  WHERE 1 = 1" \
                + "\n  -- 숫자인 데이터만 가져오기" \
                + "\n  AND D_PALDANG REGEXP '^[0-9]+\\.?[0-9]*$'" \
                + "\n  AND D_36T REGEXP '^[0-9]+\\.?[0-9]*$'" \
                + "\n  AND D_GWANGAM REGEXP '^[0-9]+\\.?[0-9]*$'" \
                + "\n  AND D_SEONGSAN_GIMPO REGEXP '^[0-9]+\\.?[0-9]*$'" \
                + "\n  AND D_YANGJECHEON REGEXP '^[0-9]+\\.?[0-9]*$'" \
                + "\n  AND TIMESTAMP BETWEEN '%s' AND '%s'" % (time_start, time_end)

    return query

def main(location, func, time_start, time_end):
    try:
        con = conn.getConnection()
        con.set_character_set('utf8')
        cur = con.cursor(MySQLdb.cursors.DictCursor)
        cur.execute('SET NAMES utf8;')
        cur.execute('SET CHARACTER SET utf8;')
        cur.execute('SET character_set_connection=utf8;')

        result = 0

        query = getQuery(location, func)

        cur.execute(query)
        result = cur.fetchall()
        if cur.rowcount != 0 and result[0]['AVG'] is not None:
            result = int(result[0]['AVG'])

        #print('query:', query)
        print(func ,':', result)

    except Exception as e:
        print('Ex :')
        traceback.print_exc()
        pass

if __name__ == '__main__':
    # hongtong, seongsan_gimpo
    location = 'seongsan_gimpo'
    # AVG, STDDEV
    func = 'AVG'
    time_start = '2017-06-01 00:01:00'
    time_end = '2017-08-11 00:00:00'
    main(location, func, time_start, time_end)








