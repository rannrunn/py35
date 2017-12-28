import numpy as np
import pandas as pd
import os
import pymysql


class IoT_Save:

    conn = pymysql.connect(host='localhost', user='root', password='1111', db='kepco', charset='utf8')
    curs = conn.cursor(pymysql.cursors.DictCursor)  # DictCursor: row 결과를 dictionary 형태로 반환

    @staticmethod
    def savePoleId():
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
        query_select= """select distinct(pole_id) from tb_iot_pole 
                      where pole_id !='' and pole_id not like '%Test%' 
                      order by pole_id"""
        IoT_Save.curs.execute(query_select)
        result = IoT_Save.curs.fetchall()  # fetchall: data를 한번에 가져옴, fetchone: 하나의 row만 가져옴, fetchmany(n): n개만큼의 데이타를 가져옴
        data = pd.DataFrame(result)
        os.chdir("D:\\dev\\IoT_data")
        data.to_csv('2차기간_pole_id.csv', encoding='utf-8')

    @staticmethod
    def savePoleData(pole_id):
        # query_select = """select pole_id, time_id, sensor_id, ri, pi, temp, humi, pitch, roll, ambient, uv, press, battery, period, current, shock, geomag_x, geomag_y, geomag_z, var_x, var_y, var_z, usn, ntc, uvc, part_name from(
        #                 select pole_id, time_id, sensor_id, ri, pi, temp, humi, pitch, roll, ambient, uv, press, battery, period, current, shock, geomag_x, geomag_y, geomag_z, var_x, var_y, var_z, usn, ntc, uvc, NULL as part_name from tb_iot_pole
        #                 union all
        #                 select pole_id, time_id, sensor_id, ri, pi, temp, humi, pitch, roll, ambient, uv, press, battery, period, current, shock, geomag_x, geomag_y, geomag_z, var_x, var_y, var_z, usn, ntc, uvc, part_name from tb_iot_pole_201604
        #                 union all
        #                 select pole_id, time_id, sensor_id, ri, pi, temp, humi, pitch, roll, ambient, uv, press, battery, period, current, shock, geomag_x, geomag_y, geomag_z, var_x, var_y, var_z, usn, ntc, uvc, part_name from tb_iot_pole_201605
        #                 union all
        #                 select pole_id, time_id, sensor_id, ri, pi, temp, humi, pitch, roll, ambient, uv, press, battery, period, current, shock, geomag_x, geomag_y, geomag_z, var_x, var_y, var_z, usn, ntc, uvc, part_name from tb_iot_pole_201606
        #                 union all
        #                 select pole_id, time_id, sensor_id, ri, pi, temp, humi, pitch, roll, ambient, uv, press, battery, period, current, shock, geomag_x, geomag_y, geomag_z, var_x, var_y, var_z, usn, ntc, uvc, part_name from tb_iot_pole_201607
        #                 union all
        #                 select pole_id, time_id, sensor_id, ri, pi, temp, humi, pitch, roll, ambient, uv, press, battery, period, current, shock, geomag_x, geomag_y, geomag_z, var_x, var_y, var_z, usn, ntc, uvc, part_name from tb_iot_pole_201608
        #                 union all
        #                 select pole_id, time_id, sensor_id, ri, pi, temp, humi, pitch, roll, ambient, uv, press, battery, period, current, shock, geomag_x, geomag_y, geomag_z, var_x, var_y, var_z, usn, ntc, uvc, part_name from tb_iot_pole_201609
        #                 union all
        #                 select pole_id, time_id, sensor_id, ri, pi, temp, humi, pitch, roll, ambient, uv, press, battery, period, current, shock, geomag_x, geomag_y, geomag_z, var_x, var_y, var_z, usn, ntc, uvc, part_name from tb_iot_pole_201610
        #                 union all
        #                 select pole_id, time_id, sensor_id, ri, pi, temp, humi, pitch, roll, ambient, uv, press, battery, period, current, shock, geomag_x, geomag_y, geomag_z, var_x, var_y, var_z, usn, ntc, uvc, part_name from tb_iot_pole_201611
        #                 union all
        #                 select pole_id, time_id, sensor_id, ri, pi, temp, humi, pitch, roll, ambient, uv, press, battery, period, current, shock, geomag_x, geomag_y, geomag_z, var_x, var_y, var_z, usn, ntc, uvc, part_name from tb_iot_pole_201612
        #                 ) a
        #                 where pole_id='%s'
        #                 order by sensor_id, time_id""" % (pole_id)
        # query_select = """select * from tb_iot_pole
        #                 where pole_id='%s' and time_id < '2017-06-19'
        #                 order by sensor_id, time_id""" % (pole_id)
        query_select = """select * from(
                        select * from tb_iot_pole_201604
                        union all
                        select * from tb_iot_pole_201605
                        union all
                        select * from tb_iot_pole_201606
                        union all
                        select * from tb_iot_pole_201607
                        union all
                        select * from tb_iot_pole_201608
                        union all
                        select * from tb_iot_pole_201609
                        union all
                        select * from tb_iot_pole_201610
                        union all
                        select * from tb_iot_pole_201611
                        union all
                        select * from tb_iot_pole_201612
                        ) a
                        where pole_id='%s'
                        order by sensor_id, time_id""" % (pole_id)
        IoT_Save.curs.execute(query_select)
        result = IoT_Save.curs.fetchall()
        df = pd.DataFrame(result)
        # 저장 경로 지정
        os.chdir("D:\\dev\\IoT_data\\IoT_233")
        df.to_csv(pole_id + '.csv', encoding='utf-8')


if __name__ == '__main__':
    IoT_Save.savePoleId()
    os.chdir("D:\\dev\\IoT_data")
    poleId = pd.read_csv('pole_id_233.csv')
    # iot=IoT_Save
    for pole_id in poleId['pole_id']:
        print(pole_id)
        IoT_Save.savePoleData(pole_id)
    # IoT_Save.savePoleData('8132X291')

