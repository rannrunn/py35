import xmltodict
import json
import pymongo
import datetime
import os.path


connection = pymongo.MongoClient("localhost", 27017)
db = connection.kepco
collection  = db.iot

#db.iot.insert({"data_seq":"","time_id":"","sensor_id":"","pole_id":"","ri":"","pi":"","temp":"","humi":"","pitch":"","roll":"","ambient":"","uv":"","press":"","battery":"","period":"","current":"","shock":"","geomag_x":"","geomag_y":"","geomag_z":"","var_x":"","var_y":"","var_z":"","usn":"","ntc":"","uvc":""})

key_m2m = ["con","ri","pi","current","shock","geomag_x","geomag_y","geomag_z","var_x","var_y","var_z","usn","ntc","uvc"]
key_con = ["accero","temp","humi","ambient","uv","press","battery","period","current","shock","geomag_x","geomag_y","geomag_z","var_x","var_y","var_z","usn","ntc","uvc"]
key_accero = ["pitch","roll","current","shock","geomag_x","geomag_y","geomag_z","var_x","var_y","var_z","usn","ntc","uvc"]

cnt = 0
data_seq = 0
file_first = "./file/sensor_"
file_date = "2017-07-31"

while True:
    file_name = file_first + file_date
    if file_date == "2017-09-21":
        break
    elif os.path.exists(file_name):
        f = open(file_name, 'r')
        #print(file_name)
        while True:
            line = f.readline()
            if not line: break

            data_seq = data_seq + 1
            dict = {}
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
                        dict[key] = ""

            if bool_accero:
                for key in key_accero:
                    if key in json_con["accero"]:
                        dict[key] = json_con["accero"][key]
                    else:
                        dict[key] = ""

            #collection.insert(dict)

            #print(dict)

            #{"data_seq": "", "time_id": "", "sensor_id": "", "pole_id": "", "ri": "", "pi": "", "temp": "", "humi": "",
            # "pitch": "", "roll": "", "ambient": "", "uv": "", "press": "", "battery": "", "period": "", "current": "",
            # "shock": "", "geomag_x": "", "geomag_y": "", "geomag_z": "", "var_x": "", "var_y": "", "var_z": "",
            # "usn": "", "ntc": "", "uvc": ""}


            


            cnt = cnt + 1
            if cnt % 10000 == 0:
                print("file_name:", file_name, ",", "cnt:", cnt)

        f.close()

    FMT = '%Y-%m-%d'
    file_date = (datetime.datetime.strptime(file_date, FMT) + datetime.timedelta(days=1)).strftime(FMT)



