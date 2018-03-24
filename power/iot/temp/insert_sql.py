# MariaDB 에 파일을 insert하기 위한 소스
# coding: utf-8

import xmltodict
import json
import datetime
import os.path
import MySQLdb
import time
import copy
from multiprocessing import Pool

start = time.time()


# DB insert 구문
query_insert = 'INSERT IGNORE INTO `NT_SENSOR_XML_TEST` (`SENSOR_ID`, `SENSOR_OID`, `DATA_GROUP_NO`, `XML_DATA`, `INS_ID`, `INS_DT`) VALUES (%s,%s,%s,%s,%s,%s)'
print(query_insert)


# DB 에 여러개 데이터를 한번에 인서트 하기 위한 list 생성
def make_values_list(values_list, tup):
    values_list.append(tup)
    return values_list

# excutemany를 이용해 한번에 insert
def insert_execute(con, cur, values_list):
    cur.executemany(query_insert, values_list)
    con.commit()

#db.iot.insert({"data_seq":"","time_id":"","sensor_id":"","pole_id":"","ri":"","pi":"","temp":"","humi":"","pitch":"","roll":"","ambient":"","uv":"","press":"","battery":"","period":"","current":"","shock":"","geomag_x":"","geomag_y":"","geomag_z":"","var_x":"","var_y":"","var_z":"","usn":"","ntc":"","uvc":""})

def parser(file_path):

    if not os.path.exists(file_path):
        return

    # MariaDB connection
    con = MySQLdb.connect('192.168.0.7', 'root', '1111', 'KEPIOT')
    con.set_character_set('utf8')
    cur = con.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SET NAMES utf8;')
    cur.execute('SET CHARACTER SET utf8;')
    cur.execute('SET character_set_connection=utf8;')

    f = open(file_path, 'rt', encoding='UTF8')
    #print(file_name)

    cnt = 0
    values_list = []

    while True:

        line = f.readline()
        if not line:
            break

        # 라인이 존재하는지 판단한 후에 카운트를 시작한다.
        cnt += 1

        bool_con = False
        bool_accero = False

        if line.find('(') < 0 or line.find('(') > 10:
            continue


        # XML 이전에 설정된 세 개의 키를 split
        line_split = line[line.find('(') + 1:line.find(')')].split(',')
        tup = tuple([line_split[0].strip(), line_split[1].strip(), line_split[2].strip(), line_split[3].strip(), line_split[4].strip(), line_split[5].strip()])

        values_list = make_values_list(values_list, tup)

        #print(line_split)

        if cnt % 100000 == 0:
            insert_execute(con, cur, values_list)
            values_list = []
            if cnt % 1000000 == 0:
                print("cnt:", cnt)
                print('iot_parser_maria : Past time : %f' % (time.time() - start))

    # 마지막 파일일 경우에 아직 INSERT 되지 않는 데이터에 대해 INSERT
    if cnt % 10000 != 0:
        insert_execute(con, cur, values_list)
        pass

    f.close()
    con.close()



if __name__ == '__main__':

    path = "D:/KEPIOT/KEPIOT"
    list_file = []
    list_file.append('/'.join([path, 'NT_SENSOR_XML.sql']))

    parser(list_file[0])








