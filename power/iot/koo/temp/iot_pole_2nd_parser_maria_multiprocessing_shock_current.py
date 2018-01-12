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
key_accero = ["pitch","roll","current","shock","accero","temp","humi","ambient","uv","press","battery","period","current","shock","geomag_x","geomag_y","geomag_z","var_x","var_y","var_z","usn","ntc","uvc","accero","temp","humi","ambient","uv","press","battery","period","current","shock","geomag_x","geomag_y","geomag_z","var_x","var_y","var_z","usn","ntc","uvc"]


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

    # MariaDB connection
    con = MySQLdb.connect('localhost', 'root', '1111', 'kepco')
    cur = con.cursor(MySQLdb.cursors.DictCursor)

    f = open(file_path, 'r')
    #print(file_name)

    cnt = 0
    values_list = []



    while True:
        line = f.readline()
        if not line:
            break

        ccc = 0

        # 라인이 존재하는지 판단한 후에 카운트를 시작한다.
        cnt += 1

        if cnt % 50000 == 0:
            print('file:{},cnt:{}'.format(file_path, cnt))

        dict = copy.deepcopy(dict_initial)

        bool_con = False
        bool_accero = False

        # XML 이전에 설정된 세 개의 키를 split
        line_split = line[:line.index('<')].split(',')

        # XML 데이터를 JSON으로 파싱
        line_xml = line[line.index('<'):]
        root = xmltodict.parse(line_xml)
        resp = json.dumps(root)
        json_all = json.loads(resp)

        # XML 루트 검색
        for key in json_all["m2m:cin"]:

            if key == "con":
                bool_con = True



        #print(json_all["m2m:cin"]["con"])
        # 형식을 벗어나는 데이터 수정
        json_con = json.loads(json_all["m2m:cin"]["con"].replace("\"\"temp\"","\",\"temp\""))

        # con 이 있을 경우 탐색
        if bool_con:
            for key in json_con:


                if key == "accero":
                    bool_accero = True



        # accero가 있을 경우 탐색
        if bool_accero:
            for key in json_con["accero"]:

                if key not in list_accero:

                    list_accero.append(key)

    print('iot_parser_maria : Total time : %f' % (time.time() - start))


    f.close()
    con.close()



if __name__ == '__main__':


    path = "D:/010_data/kepco/iot_pole/2nd/data"
    list_file = []
    list_file = ['D:/010_data/kepco/iot_pole/2nd/data/sensor_2017-08-09']
    for path, dir, filenames in os.walk(path):
        for item in filenames:
            parser('/'.join([path, item]))

    # list_file = ['D:/010_data/kepco/iot_pole/2nd/data/sensor_2017-08-09']




