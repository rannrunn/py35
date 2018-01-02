# MongoDB에 파일 insert하기 위한 소스

import xmltodict
import json
import pymongo
import datetime
import os.path

# MongoDB connection
connection = pymongo.MongoClient("localhost", 27017)
db = connection.kepco
collection  = db.iot

#db.iot.insert({"data_seq":"","time_id":"","sensor_id":"","pole_id":"","ri":"","pi":"","temp":"","humi":"","pitch":"","roll":"","ambient":"","uv":"","press":"","battery":"","period":"","current":"","shock":"","geomag_x":"","geomag_y":"","geomag_z":"","var_x":"","var_y":"","var_z":"","usn":"","ntc":"","uvc":""})

# XML 관련 키
key_m2m = ["con","ri","pi","current","shock","geomag_x","geomag_y","geomag_z","var_x","var_y","var_z","usn","ntc","uvc"]
key_con = ["accero","temp","humi","ambient","uv","press","battery","period","current","shock","geomag_x","geomag_y","geomag_z","var_x","var_y","var_z","usn","ntc","uvc"]
key_accero = ["pitch","roll","current","shock","geomag_x","geomag_y","geomag_z","var_x","var_y","var_z","usn","ntc","uvc"]

# 카운트 초기화
cnt = 0
file_first = "./file/sensor_"
# 파일을 초기 날짜 설정
file_date = "2017-07-31"

while True:
    file_name = file_first + file_date
    # 검색을 끝낼 날짜 설정
    if file_date == "2017-09-21":
        break
    elif os.path.exists(file_name):
        f = open(file_name, 'r')
        #print(file_name)
        while True:
            line = f.readline()
            if not line: break

            dict = {}
            # XML에서 con 의 존재 여부
            bool_con = False
            # XML에서 accero의 존재 여부
            bool_accero = False

            # XML 이전에 설정된 세개의 키를 split
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

            #collection.insert(dict)

            #print(dict)

            cnt = cnt + 1
            if cnt % 10000 == 0:
                print("file_name:", file_name, ",", "cnt:", cnt)

        f.close()

    # 파일의 날짜를 하루 더함
    FMT = '%Y-%m-%d'
    file_date = (datetime.datetime.strptime(file_date, FMT) + datetime.timedelta(days=1)).strftime(FMT)



