import MySQLdb
import os.path

file_first = "C:/_data/data/sensor_"
file_date = "2017-06-20"
file_name = file_first + file_date

if os.path.exists(file_name):
    f = open(file_name, 'r')
    while True:
        line = f.readline()
        if not line: break;
        line_xml = line[line.index('<'):]

        #json_con = json.loads(json_all["m2m:cin"]["con"].replace("\"\"temp\"","\",\"temp\""))