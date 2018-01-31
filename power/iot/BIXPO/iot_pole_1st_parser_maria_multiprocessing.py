# 2차 파일 들을 멀티코어를 사용하여 insert 하기 위한 소스
# sensor_xml_data_20170515 자료에 맞게 iot_pole_2nd_parser_maria.py 소스를 수정
# sensor_xml_data_20170515 에 들어 있는 자료의 키는 모두 5개이다
# coding: utf-8
import xmltodict
import json
import datetime
import os.path
import MySQLdb
import time
import traceback
import threading
import logging
import os
from multiprocessing import Process

# DB 에 여러개 데이터를 한번에 인서트 하기 위한 list 생성
def make_values_list(values_list, dict):
    values_list.append(tuple(dict.values()))
    return values_list

# 한번에 insert
def insert_execute(con, cur, query_insert, values_list):
    cur.executemany(query_insert, values_list)
    con.commit()
    # 인서트 마쳤으니 초기화
    values_list = []
    return values_list

def parser(name):

    # MariaDB connection
    con = MySQLdb.connect('localhost', 'root', '1111', 'kepco')
    con.set_character_set('utf8')
    cur = con.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SET NAMES utf8;')
    cur.execute('SET CHARACTER SET utf8;')
    cur.execute('SET character_set_connection=utf8;')

    start = time.time()

    # XML 주요 키 세 가지와 그의 서브 키
    key_m2m = ["con","ri","pi"]
    key_con = ["accero","temp","humi","pitch","roll","ambient","uv","press","battery","period","geomag_x","geomag_y","geomag_z","var_x","var_y","var_z","usn","ntc","uvc","current","shock"]
    key_accero = ["pitch","roll","current","shock"]

    cnt = 0
    cnt_batch = 10000
    values_list = []

    # dictionary 초기화
    dict_initial = {'file_name':'','time_id':'','sensor_id':'','pole_id':'','part_name':'','ri':'','pi':'','temp':'','humi':'','pitch':'','roll':'','ambient':'','uv':'','press':'','battery':'','period':'','current':'','shock':'','geomag_x':'','geomag_y':'','geomag_z':'','var_x':'','var_y':'','var_z':'','usn':'','ntc':'','uvc':''}
    columns = ','.join(dict_initial.keys())

    # 파일명을 변경해 가면서 인서트
    file_name = 'D:/010_data/kepco/iot_pole/1st/'+ name

    if not os.path.exists(file_name):
        return

    # 테이블명
    table = 'tb_iot_pole_' + name[13:19]
    query_insert = "insert into " + table + "(" + columns + ") values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"

    # dictionary에 파일명 추가
    dict_initial['file_name'] = os.path.basename(file_name)

    f = open(file_name, 'rt', encoding='UTF8')
    #print(file_name)
    while True:

        line = f.readline()
        if not line:
            # 마지막 라인일 경우에 아직 INSERT 되지 않는 데이터에 대해 INSERT
            if cnt % cnt_batch != 0:
                values_list = insert_execute(con, cur, query_insert, values_list)
                print("file_name:", file_name, " , cnt:", cnt, " , past_time:", (time.time() - start))
            break

        dict = dict_initial

        bool_con = False
        bool_accero = False

        line_split = line.split(',')

        # XML 이전에 설정된 네 개의 키를 split
        dict["time_id"] = line_split[3].replace('"',"")
        dict["sensor_id"] = line_split[0].replace('"',"")
        dict["pole_id"] = line_split[1].replace('"',"")
        dict["part_name"] = line_split[2].replace('"',"")

        # 데이터 내용 중 형식에 맞지 않는 내용 수정
        line_xml = line[line.index('<'):].replace('</m2m:cin>"', '</m2m:cin>')
        line_xml = line_xml.replace('""', '"')
        line_xml = line_xml.replace(',,', ',')
        line_xml = line_xml.replace(',}', '}')

        # XML 데이터를 JSON으로 파싱
        root = xmltodict.parse(line_xml)
        resp = json.dumps(root)
        json_all = json.loads(resp)

        # XML 루트 검색
        for key in json_all["m2m:cin"]:
            if key in key_m2m:
                if key == "con":
                    bool_con = True
                else:
                    dict[key] = json_all["m2m:cin"][key]

        #print(json_all["m2m:cin"]["con"])
        json_con = {}
        if bool(json_all["m2m:cin"]["con"]):
            json_con = json.loads(json_all["m2m:cin"]["con"].replace("\"\"temp\"","\",\"temp\""))

        # con 이 있을 경우 탐색
        if bool_con:
            for key in json_con:
                if key in key_con:
                    if key == "accero":
                        bool_accero = True
                    else:
                        dict[key] = json_con[key]

        # accero가 있을 경우 탐색
        if bool_accero:
            for key in json_con["accero"]:
                if key in key_accero:
                    dict[key] = json_con["accero"][key]

        #print(dict)
        # 한번에 insert할 리스트 생성
        values_list = make_values_list(values_list, dict)

        # cnt_batch의 개수에 도달했을 때 한꺼번에 insert
        cnt = cnt + 1
        if cnt % cnt_batch == 0:
            values_list = insert_execute(con, cur, query_insert, values_list)
            print("file_name:", file_name, " , cnt:", cnt, " , past_time:", (time.time() - start))

    f.close()
    con.close()


if __name__ == '__main__':

    # names = ['t_SENSOR_XML_201604_test.dat']
    # insert할 파일의 리스트
    names = ['t_SENSOR_XML_201604_2274787.dat'
        ,'t_SENSOR_XML_201605_1895870.dat'
        ,'t_SENSOR_XML_201606_1371773.dat'
        ,'t_SENSOR_XML_201607_4064409.dat'
        ,'t_SENSOR_XML_201608_8218811.dat'
        ,'t_SENSOR_XML_201609_7187682.dat'
        ,'t_SENSOR_XML_201610_6699561.dat'
        ,'t_SENSOR_XML_201611_5084265.dat'
        ,'t_SENSOR_XML_201612_20170515_30571302.txt']

    # 멀티스로세싱
    procs = []
    for name in names:
        proc = Process(target=parser, args=(name,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
