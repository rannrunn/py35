import xmltodict
import json
import datetime
import os.path
import MySQLdb
import time

start = time.time()

con = MySQLdb.connect('localhost', 'root', '1111', 'kepco')
cur = con.cursor(MySQLdb.cursors.DictCursor)
table = 'TB_IOT_POLE'
dict_initial = {'time_id':'','sensor_id':'','pole_id':'','ri':'','pi':'','temp':'','humi':'','pitch':'','roll':'','ambient':'','uv':'','press':'','battery':'','period':'','current':'','shock':'','geomag_x':'','geomag_y':'','geomag_z':'','var_x':'','var_y':'','var_z':'','usn':'','ntc':'','uvc':''}
columns = ','.join(dict_initial.keys())
query_insert = 'insert into TB_IOT_POLE(' + columns + ') values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
print(query_insert)

def select(table):
    query = 'select * from %s' % (table)
    cur.execute(query);
    results = cur.fetchall()
    for row in results:
        print(row)

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
    values_list = []

    while True:
        file_name = file_first + file_date
        if file_date == "2017-09-21":
            break
        elif os.path.exists(file_name):
            f = open(file_name, 'r')
            #print(file_name)
            while True:
                line = f.readline()
                if not line:
                    if cnt % 10000 != 0 and cnt > 27530000:
                        insert_execute(values_list)
                    break

                if cnt >= 27530000:

                    dict = dict_initial

                    bool_con = False
                    bool_accero = False

                    line_split = line[:line.index('<')].split(',')
                    dict["time_id"] = line_split[2]
                    dict["sensor_id"] = line_split[0]
                    dict["pole_id"] = line_split[1]

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
                    #insert_execute(values_list)
                    values_list = []
                    print("file_name:", file_name, ",", "cnt:", cnt)
                    print('iot_parser_maria : Total time : %f' % (time.time() - start))

            f.close()

        FMT = '%Y-%m-%d'
        file_date = (datetime.datetime.strptime(file_date, FMT) + datetime.timedelta(days=1)).strftime(FMT)


if __name__ == '__main__':
    main()