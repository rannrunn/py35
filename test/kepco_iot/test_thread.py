# coding: utf-8
# sensor_xml_data_20170515 자료에 맞게 소스 수정
# sensor_xml_data_20170515 에 들어 있는 자료의 키는 모두 5개이다
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

logging.basicConfig(level=logging.DEBUG, format='(%(threadName)-10s) %(message)s', )

def make_values_list(values_list, dict):
    values_list.append(tuple(dict.values()))
    return values_list

def insert_execute(con, cur, query_insert, values_list):
    cur.executemany(query_insert, values_list)
    con.commit()
    # 인서트 마쳤으니 초기화
    values_list = []
    return values_list

def parser(name):

    con = MySQLdb.connect('localhost', 'root', '1111', 'kepco')
    con.set_character_set('utf8')
    cur = con.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SET NAMES utf8;')
    cur.execute('SET CHARACTER SET utf8;')
    cur.execute('SET character_set_connection=utf8;')

    start = time.time()

    key_m2m = ["con","ri","pi","current","shock"]
    key_con = ["accero","temp","humi","ambient","uv","press","battery","period","geomag_x","geomag_y","geomag_z","var_x","var_y","var_z","usn","ntc","uvc","current","shock"]
    key_accero = ["pitch","roll","current","shock"]

    cnt = 0
    values_list = []

    dict_initial = {'file_name':'','time_id':'','sensor_id':'','pole_id':'','part_name':'','ri':'','pi':'','temp':'','humi':'','pitch':'','roll':'','ambient':'','uv':'','press':'','battery':'','period':'','current':'','shock':'','geomag_x':'','geomag_y':'','geomag_z':'','var_x':'','var_y':'','var_z':'','usn':'','ntc':'','uvc':''}
    columns = ','.join(dict_initial.keys())

    # 파일명을 변경해 가면서 인서트
    file_name = 'E:/sensor_xml_data_20170515.tar/sensor_xml_data_20170515/'+ name

    if not os.path.exists(file_name):
        return

    table = 'tb_iot_pole_' + name[13:19]
    query_insert = "insert into " + table + "(" + columns + ") values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"

    dict_initial['file_name'] = os.path.basename(file_name)

    f = open(file_name, 'rt', encoding='UTF8')
    #print(file_name)
    while True:

        line = f.readline()
        if not line:
            # 마지막 라인일 경우에 아직 INSERT 되지 않는 데이터에 대해 INSERT
            if cnt % 10000 != 0:
                values_list = insert_execute(con, cur, query_insert, values_list)
                print("file_name:", file_name, ",", "cnt:", cnt)
                print('iot_parser_maria : Total time : %f' % (time.time() - start))
            break

        dict = dict_initial

        bool_con = False
        bool_accero = False

        line_split = line.split(',')

        dict["time_id"] = line_split[3].replace('"',"")
        dict["sensor_id"] = line_split[0].replace('"',"")
        dict["pole_id"] = line_split[1].replace('"',"")
        dict["part_name"] = line_split[2].replace('"',"")

        line_xml = line[line.index('<'):].replace('</m2m:cin>"', '</m2m:cin>')
        line_xml = line_xml.replace('""', '"')
        line_xml = line_xml.replace(',,', ',')

        root = xmltodict.parse(line_xml)
        resp = json.dumps(root)
        json_all = json.loads(resp)

        for key in key_m2m:
            if key in json_all["m2m:cin"]:
                if key == "con":
                    bool_con = True
                else:
                    dict[key] = json_all["m2m:cin"][key]

        #print(json_all["m2m:cin"]["con"])
        json_con = {}
        if bool(json_all["m2m:cin"]["con"]):
            json_con = json.loads(json_all["m2m:cin"]["con"].replace("\"\"temp\"","\",\"temp\""))

        if bool_con:
            for key in key_con:
                if key in json_con:
                    if key == "accero":
                        bool_accero = True
                    else:
                        dict[key] = json_con[key]

        if bool_accero:
            for key in key_accero:
                if key in json_con["accero"]:
                    dict[key] = json_con["accero"][key]

        #print(dict)

        values_list = make_values_list(values_list, dict)

        cnt = cnt + 1
        if cnt % 10000 == 0:
            values_list = insert_execute(con, cur, query_insert, values_list)
            print("file_name:", file_name, " , cnt:", cnt, " , past_time:", (time.time() - start))

    f.close()


if __name__ == '__main__':

    names = ['t_SENSOR_XML_201604_2274787.dat'
        ,'t_SENSOR_XML_201605_1895870.dat'
        ,'t_SENSOR_XML_201606_1371773.dat'
        ,'t_SENSOR_XML_201607_4064409.dat'
        ,'t_SENSOR_XML_201608_8218811.dat'
        ,'t_SENSOR_XML_201609_7187682.dat'
        ,'t_SENSOR_XML_201610_6699561.dat'
        ,'t_SENSOR_XML_201611_5084265.dat'
        ,'t_SENSOR_XML_201612_20170515_30571302.txt']

    procs = []

    for name in names:
        proc = Process(target=parser, args=(name,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
