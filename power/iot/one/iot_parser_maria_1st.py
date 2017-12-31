# MariaDB 에 파일을 insert하기 위한 소스

import xmltodict
import json
import datetime
import os.path
import MySQLdb
import time

start = time.time()

# MariaDB connection
con = MySQLdb.connect('localhost', 'root', '1111', 'kepco')
cur = con.cursor(MySQLdb.cursors.DictCursor)
# 테이블 명
table = 'TB_IOT_POLE_V3'
# DB 에 insert 하기 위한 dictionary 형식
dict_initial = {'file_name':'','time_id':'','sensor_id':'','pole_id':'','ri':'','pi':'','temp':'','humi':'','pitch':'','roll':'','ambient':'','uv':'','press':'','battery':'','period':'','current':'','shock':'','geomag_x':'','geomag_y':'','geomag_z':'','var_x':'','var_y':'','var_z':'','usn':'','ntc':'','uvc':''}
# 컬럼 생성
columns = ','.join(dict_initial.keys())
# DB insert 구문
query_insert = 'insert into ' + table + '(' + columns + ') values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
print(query_insert)

# DB 에 여러개 데이터를 한번에 인서트 하기 위한 list 생성
def make_values_list(values_list, dict):
    values_list.append(tuple(dict.values()))
    return values_list

# excutemany를 이용해 한번에 insert
def insert_execute(values_list):
    cur.executemany(query_insert, values_list)
    con.commit()

#db.iot.insert({"data_seq":"","time_id":"","sensor_id":"","pole_id":"","ri":"","pi":"","temp":"","humi":"","pitch":"","roll":"","ambient":"","uv":"","press":"","battery":"","period":"","current":"","shock":"","geomag_x":"","geomag_y":"","geomag_z":"","var_x":"","var_y":"","var_z":"","usn":"","ntc":"","uvc":""})

def main():

    # XML 주요 키 세 가지와 그의 서브 키
    key_m2m = ["con","ri","pi","current","shock"]
    key_con = ["accero","temp","humi","ambient","uv","press","battery","period","current","shock","geomag_x","geomag_y","geomag_z","var_x","var_y","var_z","usn","ntc","uvc"]
    key_accero = ["pitch","roll","current","shock"]

    cnt = 0
    file_first = "C:/_data/data/sensor_"
    # insert 할 파일 시작일
    file_date = "2017-06-20"
    # insert 할 파일 종료일
    last_date = "2017-09-20"
    FMT = '%Y-%m-%d'
    values_list = []

    while True:
        file_name = file_first + file_date
        if file_date == (datetime.datetime.strptime(last_date, FMT) + datetime.timedelta(days=1)).strftime(FMT):
            break
        elif os.path.exists(file_name):
            f = open(file_name, 'r')
            #print(file_name)
            while True:
                line = f.readline()
                if not line:
                    # 마지막 파일일 경우에 아직 INSERT 되지 않는 데이터에 대해 INSERT
                    if cnt % 10000 != 0 and file_date == last_date:
                        insert_execute(values_list)
                        print("last_file_name:", file_name, ",", "cnt:", cnt)
                    break

                dict = dict_initial

                bool_con = False
                bool_accero = False

                dict["file_name"] = os.path.basename(file_name)

                # XML 이전에 설정된 세 개의 키를 split
                line_split = line[:line.index('<')].split(',')
                dict["time_id"] = line_split[2]
                dict["sensor_id"] = line_split[0]
                dict["pole_id"] = line_split[1]

                # XML 데이터를 JSON으로 파싱
                line_xml = line[line.index('<'):]
                root = xmltodict.parse(line_xml)
                resp = json.dumps(root)
                json_all = json.loads(resp)

                # XML 루트 검색
                for key in key_m2m:
                    if key in json_all["m2m:cin"]:
                        if key == "con":
                            bool_con = True
                        else:
                            dict[key] = json_all["m2m:cin"][key]
                    else:
                        if key != "con":
                            dict[key] = ""

                #print(json_all["m2m:cin"]["con"])
                # 형식을 벗어나는 데이터 수정
                json_con = json.loads(json_all["m2m:cin"]["con"].replace("\"\"temp\"","\",\"temp\""))

                # con 이 있을 경우 탐색
                if bool_con:
                    for key in key_con:
                        if key in json_con:
                            if key == "accero":
                                bool_accero = True
                            else:
                                dict[key] = json_con[key]
                        else:
                            if key != "accero":
                                dict[key] = ""

                # accero가 있을 경우 탐색
                if bool_accero:
                    for key in key_accero:
                        if key in json_con["accero"]:
                            dict[key] = json_con["accero"][key]
                        else:
                            dict[key] = ""

                #print(dict)

                values_list = make_values_list(values_list, dict)

                cnt = cnt + 1
                if cnt % 10000 == 0:
                    insert_execute(values_list)
                    values_list = []
                    print("file_name:", file_name, ",", "cnt:", cnt)
                    print('iot_parser_maria : Total time : %f' % (time.time() - start))

            f.close()

        # 파일의 날짜를 하루 더함
        file_date = (datetime.datetime.strptime(file_date, FMT) + datetime.timedelta(days=1)).strftime(FMT)


if __name__ == '__main__':
    main()