import xmltodict
import json
import datetime
import os.path
import MySQLdb
import time
import traceback

key_m2m = ["con","ri","pi","current","shock"]
key_con = ["accero","temp","humi","ambient","uv","press","battery","period","geomag_x","geomag_y","geomag_z","var_x","var_y","var_z","usn","ntc","uvc","current","shock"]
key_accero = ["pitch","roll","current","shock"]

dict = {'file_name':'','time_id':'','sensor_id':'','pole_id':'','part_name':'','ri':'','pi':'','temp':'','humi':'','pitch':'','roll':'','ambient':'','uv':'','press':'','battery':'','period':'','current':'','shock':'','geomag_x':'','geomag_y':'','geomag_z':'','var_x':'','var_y':'','var_z':'','usn':'','ntc':'','uvc':''}

line_xml = '<?xml version=""1.0"" encoding=""UTF-8"" standalone=""yes""?><m2m:cin xmlns:m2m=""http://www.onem2m.org/xml/protocols""><ty>4</ty><ri>CI00000000000078527814</ri><rn>CI00000000000078527814</rn><pi>CT00000000000000001807</pi><ct>20170430T152853</ct><lt>20170430T152853</lt><lbl></lbl><at></at><aa></aa><st>3111</st><cnf></cnf><cs>419</cs><con>{""temp"":""15.0"",""alarm_temp"":""0"",""uvc"":""0.089"",""alarm_uvc"":""0"",""pitch"":""-178"",""alarm_pitch"":""0"",""roll"":""0"",""alarm_roll"":""0"",""var_x"":""58"",""var_y"":""-23"",""var_z"":""-1007"",""alarm_var_x"":""0"",""alarm_var_y"":""0"",""alarm_var_z"":""0"",""geomag_x"":""13"",""geomag_y"":""-355"",""geomag_z"":""-751"",""alarm_geomag_x"":""0"",""alarm_geomag_y"":""0"",""alarm_geomag_z"":""0"",""usn"":""0"",""alarm_usn"":""0"",""battery"":""3.02"",""alarm_battery"":""0"","""":"""",,"""":"""",,"""":"""",}</con></m2m:cin>'
line_xml = line_xml.replace('""', '"')
line_xml = line_xml.replace(',,', ',')

root = xmltodict.parse(line_xml)
resp = json.dumps(root)
json_all = json.loads(resp)

for key in key_m2m:
    if key in json_all["m2m:cin"]:
        if key == "con":
            bool_con = True
        else:
            dict[key] = json_all["m2m:cin"][key]

#print(json_all["m2m:cin"]["con"])
json_con = {}
if bool(json_all["m2m:cin"]["con"]):
    json_con = json.loads(json_all["m2m:cin"]["con"].replace("\"\"temp\"","\",\"temp\""))
    pass