import sys, os, numpy as np, pandas as pd, csv
import pymysql

# pole_id csv로 저장
# conn = pymysql.connect(host='localhost', user='root', password='1111', db='kepco', charset='utf8')
# curs = conn.cursor(pymysql.cursors.DictCursor)
# query_select = """select pole_id from
#                 (select pole_id from tb_iot_pole
#                 union select pole_id from tb_iot_pole_201604
#                 union select pole_id from tb_iot_pole_201605
#                 union select pole_id from tb_iot_pole_201606
#                 union select pole_id from tb_iot_pole_201607
#                 union select pole_id from tb_iot_pole_201608
#                 union select pole_id from tb_iot_pole_201609
#                 union select pole_id from tb_iot_pole_201610
#                 union select pole_id from tb_iot_pole_201611
#                 union select pole_id from tb_iot_pole_201612 )a
#                 where pole_id !='' and pole_id not like '%Test%'
#                 order by pole_id"""
# curs.execute(query_select)
# result = curs.fetchall()
# # print(type(result))
# data = pd.DataFrame(result)
# data.to_csv('pole_id.csv')

# pole_id 별 sensor_id, time_id, temp, humi, pitch, roll, current, shock을 csv로 저장
conn = pymysql.connect(host='localhost', user='root', password='1111', db='kepco', charset='utf8')
curs = conn.cursor(pymysql.cursors.DictCursor)
file=pd.read_csv('pole_id.csv') # pole_id.csv 불러오기
poleId=file['pole_id']
for id in poleId:
    print(id)
    query_select = """select sensor_id, time_id, temp, humi, pitch, roll, current, shock, battery, press, ambient, uv  from
                    (
                    select pole_id, sensor_id, time_id, temp, humi, pitch, roll, current, shock, battery, press, ambient, uv from tb_iot_pole
                    union all
                    select pole_id, sensor_id, time_id, temp, humi, pitch, roll, current, shock, battery, press, ambient, uv  from tb_iot_pole_201604
                    union all
                    select pole_id, sensor_id, time_id, temp, humi, pitch, roll, current, shock, battery, press, ambient, uv  from tb_iot_pole_201605
                    union all
                    select pole_id, sensor_id, time_id, temp, humi, pitch, roll, current, shock, battery, press, ambient, uv  from tb_iot_pole_201606
                    union all
                    select pole_id, sensor_id, time_id, temp, humi, pitch, roll, current, shock, battery, press, ambient, uv  from tb_iot_pole_201607
                    union all
                    select pole_id, sensor_id, time_id, temp, humi, pitch, roll, current, shock, battery, press, ambient, uv  from tb_iot_pole_201608
                    union all
                    select pole_id, sensor_id, time_id, temp, humi, pitch, roll, current, shock, battery, press, ambient, uv  from tb_iot_pole_201609
                    union all
                    select pole_id, sensor_id, time_id, temp, humi, pitch, roll, current, shock, battery, press, ambient, uv  from tb_iot_pole_201610
                    union all
                    select pole_id, sensor_id, time_id, temp, humi, pitch, roll, current, shock, battery, press, ambient, uv  from tb_iot_pole_201611
                    union all
                    select pole_id, sensor_id, time_id, temp, humi, pitch, roll, current, shock, battery, press, ambient, uv  from tb_iot_pole_201612
                    ) a
                    where pole_id='%s'
                    order by sensor_id, time_id"""%(id)
    curs.execute(query_select)
    result = curs.fetchall()
    df = pd.DataFrame(result)
    os.chdir("D:\\dev\\py_data\\pole_id_data")
    df.to_csv(id+'.csv')
