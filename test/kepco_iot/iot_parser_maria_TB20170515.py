# sensor_xml_data_20170515 자료에 맞게 소스 수정
import xmltodict
import json
import datetime
import os.path
import MySQLdb
import time

start = time.time()

con = MySQLdb.connect('localhost', 'root', '1111', 'kepco')
cur = con.cursor(MySQLdb.cursors.DictCursor)
table = 'TB_IOT_POLE_V3'
dict_initial = {'file_name':'','time_id':'','sensor_id':'','pole_id':'','part_name':'','ri':'','pi':'','temp':'','humi':'','pitch':'','roll':'','ambient':'','uv':'','press':'','battery':'','period':'','current':'','shock':'','geomag_x':'','geomag_y':'','geomag_z':'','var_x':'','var_y':'','var_z':'','usn':'','ntc':'','uvc':''}
columns = ','.join(dict_initial.keys())
query_insert = 'insert into ' + table + '(' + columns + ') values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
print(query_insert)

def make_values_list(values_list, dict):
    values_list.append(tuple(dict.values()))
    return values_list

def insert_execute(values_list):
    cur.executemany(query_insert, values_list)
    con.commit()

#db.iot.insert({"data_seq":"","time_id":"","sensor_id":"","pole_id":"","ri":"","pi":"","temp":"","humi":"","pitch":"","roll":"","ambient":"","uv":"","press":"","battery":"","period":"","current":"","shock":"","geomag_x":"","geomag_y":"","geomag_z":"","var_x":"","var_y":"","var_z":"","usn":"","ntc":"","uvc":""})

def main():
    key_m2m = ["con","ri","pi","current","shock","geomag_x","geomag_y","geomag_z","var_x","var_y","var_z","usn","ntc","uvc"]
    key_con = ["accero","temp","humi","ambient","uv","press","battery","period","current","shock","geomag_x","geomag_y","geomag_z","var_x","var_y","var_z","usn","ntc","uvc"]
    key_accero = ["pitch","roll","current","shock","geomag_x","geomag_y","geomag_z","var_x","var_y","var_z","usn","ntc","uvc"]

    cnt = 0
    file_first = "C:/_data/data/sensor_"
    file_date = "2017-06-20"
    last_date = "2017-09-20"
    FMT = '%Y-%m-%d'
    values_list = []

    while True:
        file_name = file_first + file_date

        if file_date == (datetime.datetime.strptime(last_date, FMT) + datetime.timedelta(days=1)).strftime(FMT):
            break
        if os.path.exists(file_name) == False:
            break

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

            line_split = line[:line.index('<')].split(',')
            # KEY 개수가 같을 경우 KEY 들의 위치는 달라지지 않을 것이라는 가정 하에 소스 작성
            if len(line_split) == 3:
                dict["time_id"] = line_split[2]
                dict["sensor_id"] = line_split[0]
                dict["pole_id"] = line_split[1]
            elif len(line_split) == 4:
                dict["time_id"] = line_split[3]
                dict["sensor_id"] = line_split[0]
                dict["pole_id"] = line_split[1]
                dict["part"] = line_split[2]

            line_xml = line[line.index('<'):]
            root = xmltodict.parse(line_xml)
            resp = json.dumps(root)
            json_all = json.loads(resp)

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
            json_con = json.loads(json_all["m2m:cin"]["con"].replace("\"\"temp\"","\",\"temp\""))

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

        file_date = (datetime.datetime.strptime(file_date, FMT) + datetime.timedelta(days=1)).strftime(FMT)


if __name__ == '__main__':
    main()