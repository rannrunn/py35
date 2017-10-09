import MySQLdb
import dbconnection as conn
import traceback

def main(time):
    try:
        con = conn.getConnection()
        cur = con.cursor(MySQLdb.cursors.DictCursor)

        # 공급 유량 합계 = 팔당 + 36T
        # SELECT FLOOR(D_PALDANG + D_36T) AS TOTAL_SUPPLY
        # FROM TB_WATER
        # WHERE TIMESTAMP <= '2017-06-01 00:05:00'
        # LIMIT 5
        query_one = "\n SELECT COUNT(TIMESTAMP) AS CNT" \
                  + "\n FROM TB_WATER" \
                  + "\n WHERE TIMESTAMP BETWEEN ADDTIME('%s', '-00:05:00') AND '%s'" % (time, time)

        query_two = "\n SELECT FLOOR(D_PALDANG + D_36T) AS TOTAL_SUPPLY" \
                + "\n  FROM TB_WATER" \
                + "\n  WHERE TIMESTAMP BETWEEN ADDTIME('2017-06-01 00:05:00', '-00:05:00') AND '2017-06-01 00:05:00'" \
                + "\n  LIMIT 5;"

        cur.execute(query_one)
        cnt_data = int(cur.fetchall()[0]['CNT'])
        if (cnt_data < 5):
            print('err:최근 5분 간의 데이터가 연속적이지 않거나 없는 데이터가 있습니다.')
            return

        cur.execute(query_two)
        for row in cur.fetchall():
            print(row)

        print('..')
    except Exception as e:
        print('err:Exception')
        #traceback.print_exc()
        pass

if __name__ == '__main__':
    time = '2017-06-01 00:05:00'
    main(time)




