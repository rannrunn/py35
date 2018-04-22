# MariaDB 에 파일을 insert하기 위한 소스

import xmltodict
import json
import datetime
import os.path
import MySQLdb
import time
import copy
from multiprocessing import Pool

start = time.time()

# 테이블 명
table = 'TB_IOT_POLE_SECOND'
# DB 에 insert 하기 위한 dictionary 및 list 형식
dict_initial = {'file_name':None,'time_id':None,'sensor_id':None,'pole_id':None,'ri':None,'pi':None,'temp':None,'humi':None,'pitch':None,'roll':None,'ambient':None,'uv':None,'press':None,'battery':None,'period':None,'current':None,'shock':None,'geomag_x':None,'geomag_y':None,'geomag_z':None,'var_x':None,'var_y':None,'var_z':None,'usn':None,'ntc':None,'uvc':None}
list_dict_key = ['file_name','time_id','sensor_id','pole_id','ri','pi','temp','humi','pitch','roll','ambient','uv','press','battery','period','current','shock','geomag_x','geomag_y','geomag_z','var_x','var_y','var_z','usn','ntc','uvc']
# 컬럼 생성
columns = ','.join(list_dict_key)
# DB insert 구문
query_insert = 'insert into ' + table + '(' + columns + ') values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
print(query_insert)

# XML 주요 키 세 가지와 그의 서브 키
key_m2m = ["con","ri","pi","current","shock"]
key_con = ["accero","temp","humi","ambient","uv","press","battery","period","current","shock","geomag_x","geomag_y","geomag_z","var_x","var_y","var_z","usn","ntc","uvc"]
key_accero = ["pitch","roll","current","shock"]


# DB 에 여러개 데이터를 한번에 인서트 하기 위한 list 생성
def make_values_list(values_list, dict):
    values_list.append(tuple([dict[item] for item in list_dict_key]))
    return values_list

# excutemany를 이용해 한번에 insert
def insert_execute(con, cur, values_list):
    cur.executemany(query_insert, values_list)
    con.commit()

#db.iot.insert({"data_seq":"","time_id":"","sensor_id":"","pole_id":"","ri":"","pi":"","temp":"","humi":"","pitch":"","roll":"","ambient":"","uv":"","press":"","battery":"","period":"","current":"","shock":"","geomag_x":"","geomag_y":"","geomag_z":"","var_x":"","var_y":"","var_z":"","usn":"","ntc":"","uvc":""})

def parser(file_path):

    if not os.path.exists(file_path):
        return

    list_root = []
    list_con = []
    list_accero = []
    list_ccc = []

    # MariaDB connection
    con = MySQLdb.connect('localhost', 'root', '1111', 'kepco')
    cur = con.cursor(MySQLdb.cursors.DictCursor)

    f = open(file_path, 'r', encoding='utf-8')
    #print(file_name)

    cnt = 0
    values_list = []

    ccc = 0

    while True:
        line = f.readline()
        if not line:
            break
        line = line.replace('alarm_pitch','')
        # 라인이 존재하는지 판단한 후에 카운트를 시작한다.
        cnt += 1

        if cnt % 50000 == 0:
            print('file:{},cnt:{}'.format(file_path, cnt))


        if line.count('pitch') > 1:
            print(line)
            ccc = 10000



    if ccc == 10000:
        with open('C:/csv/{}.txt'.format(os.path.basename(file_path)), 'w') as f:
            f.write('dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\n')


    f.close()
    con.close()



if __name__ == '__main__':


    names = ['t_SENSOR_XML_201612_20170515_30571302.txt']
    list_file = []
    dd = 'D:/010_data/kepco/iot_pole/1st'
    for item in names:
        parser('/'.join([dd, item]))

    # list_file = ['D:/010_data/kepco/iot_pole/2nd/data/sensor_2017-08-09']





