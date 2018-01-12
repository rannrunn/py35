# 2차 파일 들을 멀티코어를 사용하여 insert 하기 위한 소스
# sensor_xml_data_20170515 자료에 맞게 iot_pole_2nd_parser_maria.py 소스를 수정
# sensor_xml_data_20170515 에 들어 있는 자료의 키는 모두 5개이다
# coding: utf-8
import json
import os
import os.path
import time
from multiprocessing import Pool

import MySQLdb
import xmltodict
import copy

# XML 주요 키 세 가지와 그의 서브 키
key_m2m = ["con","ri","pi"]
key_con = ["accero","temp","humi","pitch","roll","ambient","uv","press","battery","period","geomag_x","geomag_y","geomag_z","var_x","var_y","var_z","usn","ntc","uvc","current","shock"]
key_accero = ["pitch","roll","current","shock"]

# dictionary 초기화
dict_initial = {'file_name':None,'time_id':None,'sensor_id':None,'pole_id':None,'part_name':None,'ri':None,'pi':None,'temp':None,'humi':None,'pitch':None,'roll':None,'ambient':None,'uv':None,'press':None,'battery':None,'period':None,'current':None,'shock':None,'geomag_x':None,'geomag_y':None,'geomag_z':None,'var_x':None,'var_y':None,'var_z':None,'usn':None,'ntc':None,'uvc':None}
list_dict_key = ['file_name','time_id','sensor_id','pole_id','part_name','ri','pi','temp','humi','pitch','roll','ambient','uv','press','battery','period','current','shock','geomag_x','geomag_y','geomag_z','var_x','var_y','var_z','usn','ntc','uvc']
columns = ','.join(list_dict_key)

# 테이블명
table = 'TB_IOT_POLE_FIRST'
query_insert = "insert into " + table + "(" + columns + ") values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"


# DB 에 여러개 데이터를 한번에 인서트 하기 위한 list 생성
def make_values_list(values_list, dict):
    values_list.append(tuple([dict[item] for item in list_dict_key]))
    return values_list

# 한번에 insert
def insert_execute(con, cur, query_insert, values_list):
    cur.executemany(query_insert, values_list)
    con.commit()

def parser(file_name):

    # MariaDB connection
    con = MySQLdb.connect('localhost', 'root', '1111', 'kepco')
    con.set_character_set('utf8')
    cur = con.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SET NAMES utf8;')
    cur.execute('SET CHARACTER SET utf8;')
    cur.execute('SET character_set_connection=utf8;')

    start = time.time()

    if not os.path.exists(file_name):
        return

    # dictionary에 파일명 추가
    dict_initial['file_name'] = os.path.basename(file_name)


    cnt = 0
    cnt_batch = 10000
    values_list = []

    f = open(file_name, 'rt', encoding='UTF8')
    #print(file_name)
    while True:

        line = f.readline()
        if not line:
            break

        cnt += 1

        dict = copy.deepcopy(dict_initial)

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

        if cnt % cnt_batch == 0:
            insert_execute(con, cur, query_insert, values_list)
            values_list = []
            print("file_name:", file_name, " , cnt:", cnt, " , past_time:", (time.time() - start))

    # 마지막 라인일 경우에 아직 INSERT 되지 않는 데이터에 대해 INSERT
    if cnt % cnt_batch != 0:
        insert_execute(con, cur, query_insert, values_list)
        print("file_name:", file_name, " , cnt:", cnt, " , Total Time:", (time.time() - start))

    f.close()
    con.close()


if __name__ == '__main__':

    path = "D:/010_data/kepco/iot_pole/1st"
    list_file = []
    for path, dir, filenames in os.walk(path):
        for item in filenames:
            list_file.append('/'.join([path, item]))

    with Pool(processes=5) as pool:
        pool.map(parser, list_file)
