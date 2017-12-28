import numpy as np
import pandas as pd
import os
import pymysql


os.chdir('D:\\dev\\IoT_data')
pole_list2=pd.read_csv('pole_test.csv')
list_pole=[]
list_data=[]
list_temp=[]
list_humi=[]
list_pitch=[]
list_roll=[]
list_ambient=[]
list_uv=[]
list_press=[]
list_battery=[]
list_current=[]
list_shock=[]
list_geomag_x=[]
list_geomag_y=[]
list_geomag_z=[]
list_var_x=[]
list_var_y=[]
list_var_z=[]
list_usn=[]
list_ntc=[]
list_uvc=[]

dict={}

for pole in pole_list2['pole_id']:
    os.chdir('D:\\dev\\IoT_data\\IoT')
    data=pd.read_csv(pole+'.csv')
    list_pole.append(pole)
    list_data.append(data.shape[0])

    list_temp.append(int((pd.isnull(data['temp']).sum()/data.shape[0])*100))
    list_humi.append(int((pd.isnull(data['humi']).sum()/data.shape[0])*100))
    list_pitch.append(int((pd.isnull(data['pitch']).sum()/data.shape[0])*100))
    list_roll.append(int((pd.isnull(data['roll']).sum()/data.shape[0])*100))
    list_ambient.append(int((pd.isnull(data['ambient']).sum()/data.shape[0])*100))
    list_uv.append(int((pd.isnull(data['uv']).sum()/data.shape[0])*100))
    list_press.append(int((pd.isnull(data['press']).sum()/data.shape[0])*100))
    list_battery.append(int((pd.isnull(data['battery']).sum()/data.shape[0])*100))
    list_current.append(int((pd.isnull(data['current']).sum()/data.shape[0])*100))
    list_shock.append(int((pd.isnull(data['shock']).sum()/data.shape[0])*100))
    list_geomag_x.append(int((pd.isnull(data['geomag_x']).sum()/data.shape[0])*100))
    list_geomag_y.append(int((pd.isnull(data['geomag_y']).sum()/data.shape[0])*100))
    list_geomag_z.append(int((pd.isnull(data['geomag_z']).sum()/data.shape[0])*100))
    list_var_x.append(int((pd.isnull(data['var_x']).sum()/data.shape[0])*100))
    list_var_y.append(int((pd.isnull(data['var_y']).sum()/data.shape[0])*100))
    list_var_z.append(int((pd.isnull(data['var_z']).sum()/data.shape[0])*100))
    list_usn.append(int((pd.isnull(data['usn']).sum()/data.shape[0])*100))
    list_ntc.append(int((pd.isnull(data['ntc']).sum()/data.shape[0])*100))
    list_uvc.append(int((pd.isnull(data['uvc']).sum()/data.shape[0])*100))


dict['pole_id']=list_pole
dict['data_num']=list_data
dict['temp']=list_temp
dict['humi']=list_humi
dict['pitch']=list_pitch
dict['roll']=list_roll
dict['ambient']=list_ambient
dict['uv']=list_uv
dict['press']=list_press
dict['battery']=list_battery
dict['current']=list_current
dict['shock']=list_shock
dict['geomag_x']=list_geomag_x
dict['geomag_y']=list_geomag_y
dict['geomag_z']=list_geomag_z
dict['var_x']=list_var_x
dict['var_y']=list_var_y
dict['var_z']=list_var_z
dict['usn']=list_usn
dict['ntc']=list_ntc
dict['uvc']=list_uvc


df=pd.DataFrame(dict)
df.to_csv('test1.csv', encoding='utf-8')

