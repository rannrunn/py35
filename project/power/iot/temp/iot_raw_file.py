# coding: utf-8
import xmltodict
import json
import datetime
import os.path
import MySQLdb
import time
import copy

file_path = 'D:/010_data/kepco/iot_pole/1st/t_SENSOR_XML_201612_20170515_30571302.txt'

dict_initial = {'file_name':'','time_id':'','sensor_id':'','pole_id':'','ri':'','pi':'','temp':'','humi':'','pitch':'','roll':'','ambient':'','uv':'','press':'','battery':'','period':'','current':'','shock':'','geomag_x':'','geomag_y':'','geomag_z':'','var_x':'','var_y':'','var_z':'','usn':'','ntc':'','uvc':''}

key_m2m = ["con","ri","pi","current","shock"]
key_con = ["accero","temp","humi","ambient","uv","press","battery","period","current","shock","geomag_x","geomag_y","geomag_z","var_x","var_y","var_z","usn","ntc","uvc"]
key_accero = ["pitch","roll","current","shock"]



with open(file_path, 'r', encoding='utf-8') as f:
    cnt = 0
    while True:
        line = f.readline()
        if not line: break

        if 'shock' in line or 'current' in line:

            print(line)














