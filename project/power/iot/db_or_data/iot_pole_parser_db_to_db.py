import json
import os.path
import time
from multiprocessing import Pool
import copy

import MySQLdb
import pymysql
import xmltodict

import re

#Default Settings
key_m2m = ["con","ri","pi"]
key_con = ["accero","pole","temp","humi","pitch","roll","ambient","uv","press","battery","period","geomag_x","geomag_y","geomag_z","var_x","var_y","var_z","usn","ntc","uvc","current","shock"]
key_accero = ["pitch","roll","current","shock"]

# dictionary 초기화
dict_initial = {'data_seq':None,'file_name':None,'time_id':None,'sensor_id':None,'sensor_oid':None,'relative_coordinate':None,'pole_id':None,'part_name':None,'ri':None,'pi':None,'temp':None,'humi':None,'pitch':None,'roll':None,'ambient':None,'uv':None,'press':None,'battery':None,'period':None,'current':None,'shock':None,'geomag_x':None,'geomag_y':None,'geomag_z':None,'var_x':None,'var_y':None,'var_z':None,'usn':None,'ntc':None,'uvc':None}
list_dict_key = ['data_seq','file_name','time_id','sensor_id','sensor_oid','relative_coordinate','pole_id','part_name','ri','pi','temp','humi','pitch','roll','ambient','uv','press','battery','period','current','shock','geomag_x','geomag_y','geomag_z','var_x','var_y','var_z','usn','ntc','uvc']
columns = ','.join(list_dict_key)
#len(list_dict_key)

print('print:', len(list_dict_key))

# 테이블명
query_insert_1 = "insert into NT_SENSOR_XML_DB_1(" + columns + ") values (%s,%s,DATE_ADD(%s, INTERVAL 9 HOUR),%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
query_insert_4 = "insert into NT_SENSOR_XML_DB_4(" + columns + ") values (%s,%s,DATE_ADD(%s, INTERVAL 9 HOUR),%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
query_insert_6 = "insert into NT_SENSOR_XML_DB_6(" + columns + ") values (%s,%s,DATE_ADD(%s, INTERVAL 9 HOUR),%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
query_insert_etc = "insert into NT_SENSOR_XML_DB_ETC(" + columns + ") values (%s,%s,DATE_ADD(%s, INTERVAL 9 HOUR),%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"


# DB 에 여러개 데이터를 한번에 인서트 하기 위한 list 생성
def make_values_list(values_list, dict):
    values_list.append(tuple([dict[item] for item in list_dict_key]))
    return values_list

def c_insert_execute(query_insert, values_list):
    pass

# 한번에 insert
def insert_execute(query_insert, values_list):
    con = MySQLdb.connect('192.168.0.7', 'root', '1111', 'SFIOT')
    con.set_character_set('utf8')
    cur = con.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SET NAMES utf8;')
    cur.execute('SET CHARACTER SET utf8;')
    cur.execute('SET character_set_connection=utf8;')
    cur.executemany(query_insert, values_list)
    con.commit()

def parser(limit_number):

    print(limit_number)
    #Connect MariaDB
    conn = pymysql.connect(host='192.168.0.7', user='root', password='1111', db='KEPIOT', charset='utf8')
    curs = conn.cursor(pymysql.cursors.DictCursor) # DictCursor: row 결과를 dictionary 형태로 반환


    num1 = (limit_number * 10000) - 4999999
    num2 = (limit_number * 10000) - 4999999 + 9999
    max_number = limit_number * 10000

    cnt = 0
    cnt_batch = 10000
    values_list_1 = []
    values_list_4 = []
    values_list_6 = []
    values_list_etc = []
    start = time.time()

    file_name = 'NT_SENSOR_XML_parser_error_log.txt'
    file_path = '/mnt/sdc1' + file_name
    out = open(file_path, 'w', encoding='UTF8')

    while num2 <= max_number:
        #Parsing test

        query_select = """select * from SFIOT.NT_SENSOR_XML_MODIFY
                       where data_seq >= %s and data_seq <= %s"""
        curs.execute(query_select, (num1, num2))

        row_cnt = 0
        print(num1, num2)

        while num1 + row_cnt <= num2:

            result = curs.fetchone() # fetchall: data를 한번에 가져옴, fetchone: 하나의 row만 가져옴, fetchmany(n): n개만큼의 데이타를 가져옴

            try:

                cnt += 1
                row_cnt += 1

                line = result['XML_DATA']

                #dict와 result 변수 mapping
                dict = copy.deepcopy(dict_initial)

                dict['data_seq'] = str(result['DATA_SEQ'])
                dict['file_name'] = None
                dict['time_id'] = result['DATA_GROUP_NO']
                dict['sensor_id'] = result['SENSOR_ID']
                dict['sensor_oid'] = result['SENSOR_OID']
                dict['relative_coordinate'] = None 
                dict["pole_id"] = None # 폴아이디가 존재하지 않는 차수가 있어 일단 빈값을 넣고 추후에 업데이트
                dict["part_name"] = None

                company_id = dict['sensor_oid'][10:11]

                bool_con = False
                bool_accero = False

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


                try:
                    str_con = json_all["m2m:cin"]["con"]
                    json_con = json.loads(str_con)
                except Exception as e:
                    if re.findall(r'"(\d+)"temp"', str_con) != None:
                        str_con = re.sub(r'"(\d+)"temp"', r'"\1","temp"', str_con)
                    if re.findall(r'"var":",', str_con) != None:
                        str_con = re.sub(r'"var":",', r'"var":"",', str_con)
                    if re.findall(r'"longitude":",', str_con) != None:
                        str_con = re.sub(r'"longitude":",', r'"longitude":"",', str_con)
                    if re.findall(r'"altitude":"}', str_con) != None:
                        str_con = re.sub(r'"altitude":"}', r'"altitude":""}', str_con)
                    if re.findall(r'"battery":",', str_con) != None:
                        str_con = re.sub(r'"battery":",', r'"battery":"",', str_con)
                    if re.findall(r'{":"}', str_con) != None:
                        str_con = re.sub(r'{":"}', r'{"":""}', str_con)
                    if re.findall(r'"latitude":",', str_con) != None:
                        str_con = re.sub(r'"latitude":",', r'"latitude":"",', str_con)

                    try:
                        json_con = json.loads(str_con)
                    except Exception as e:
                        print(str_con)
                        json_con = json.loads(str_con) 
                        

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

                # print(dict)
                # 한번에 insert할 리스트 생성
                if company_id == '1':
                    values_list_1 = make_values_list(values_list_1, dict)
                elif company_id == '4':
                    values_list_4 = make_values_list(values_list_4, dict)
                elif company_id == '6':
                    values_list_6 = make_values_list(values_list_6, dict)
                else:
                    values_list_etc = make_values_list(values_list_etc, dict)


            except Exception as e:
                pass

            finally:

                # cnt_batch의 개수에 도달했을 때 한꺼번에 insert
                if row_cnt % cnt_batch == 0:
                    insert_execute(query_insert_1, values_list_1)
                    insert_execute(query_insert_4, values_list_4)
                    insert_execute(query_insert_6, values_list_6)
                    insert_execute(query_insert_etc, values_list_etc)
                    values_list_1 = []
                    values_list_4 = []
                    values_list_6 = []
                    values_list_etc = []
                    print("cnt:", cnt, " , past_time:", (time.time() - start))


        num1 += cnt_batch
        num2 += cnt_batch

    # 마지막 라인일 경우에 아직 INSERT 되지 않는 데이터에 대해 INSERT
    if cnt % cnt_batch != 0:
        insert_execute(query_insert_1, values_list_1)
        insert_execute(query_insert_4, values_list_4)
        insert_execute(query_insert_6, values_list_6)
        insert_execute(query_insert_etc, values_list_etc)
        print("cnt:", cnt, " , Total Time:", (time.time() - start))

    conn.close()


if __name__ == '__main__':

    max_numbers = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    with Pool(processes=10) as pool:
        pool.map(parser, max_numbers)

