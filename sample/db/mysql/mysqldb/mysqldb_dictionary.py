# coding: utf8

'''
Mysql 의 Select 결과를 dictionary로 되돌려 받는 방법
'''
import MySQLdb
import MySQLdb.cursors

DB_HOST = 'localhost'
DB_USER = 'root'
DB_PWD = '1111'
DB_NAME = 'water'

if __name__ == '__main__':
    db_con = MySQLdb.connect(DB_HOST, DB_USER, DB_PWD, DB_NAME, cursorclass=MySQLdb.cursors.DictCursor)
    cursor = db_con.cursor()
    cursor.execute('SELECT * FROM TB_WATER LIMIT 10')
    result = cursor.fetchall()

    for data in result:
        for item in data.items():
            print('%s : %s'%item)
        print('=====================')

    cursor.close()
    db_con.close()
