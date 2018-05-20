# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import os
from multiprocessing import Pool
from matplotlib import font_manager, rc
import numpy as np
import time

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/맑은 고딕.ttf").get_name()
rc('font', family=font_name)

def get_list_from_location(df_sensor_info, location):

    if location == '고창':
        df_sensor_info = df_sensor_info[df_sensor_info['POLE_ADDR'].str.contains('고창군')]
    elif location == '광주':
        df_sensor_info = df_sensor_info[df_sensor_info['POLE_ADDR'].str.contains('광주 광역시') | df_sensor_info['POLE_ADDR'].str.contains('광주센서')]
    elif location == '대구':
        df_sensor_info = df_sensor_info[df_sensor_info['POLE_ADDR'].str.contains('대구 광역시') | df_sensor_info['POLE_ADDR'].str.contains('대구센서')]
    elif location == '대전':
        df_sensor_info = df_sensor_info[df_sensor_info['POLE_ADDR'].str.contains('대전 광역시') | df_sensor_info['POLE_ADDR'].str.contains('대전센서')]
    elif location == '안산':
        df_sensor_info = df_sensor_info[df_sensor_info['POLE_ADDR'].str.contains('안산센서')]

    list = df_sensor_info['FILE_NAME'].tolist()

    return list


def get_list_from_mounting_position(df_sensor_info, mounting_position):

    if mounting_position == '변압기 본체':
        df_sensor_info = df_sensor_info[df_sensor_info['MOUNTING_POSITION'].str.contains('변압기 본체')]
    elif mounting_position == '부하 개폐기':
        df_sensor_info = df_sensor_info[df_sensor_info['MOUNTING_POSITION'].str.contains('부하 개폐기')]
    elif mounting_position == '완금':
        df_sensor_info = df_sensor_info[df_sensor_info['MOUNTING_POSITION'].str.contains('완금')]
    elif mounting_position == '전주':
        df_sensor_info = df_sensor_info[df_sensor_info['MOUNTING_POSITION'].str.contains('전주')]
    elif mounting_position == '통신용 함체':
        df_sensor_info = df_sensor_info[df_sensor_info['MOUNTING_POSITION'].str.contains('통신용 함체')]

    list = df_sensor_info['SENSOR_OID'].tolist()

    return list

def get_list_from_manufacturer(df_sensor_info, manufacturer_number):

    if manufacturer_number == '1':
        df_sensor_info = df_sensor_info[df_sensor_info['SENSOR_OID'].str.slice(10,11) == '1']
    elif manufacturer_number == '4':
        df_sensor_info = df_sensor_info[df_sensor_info['SENSOR_OID'].str.slice(10,11) == '4']
    elif manufacturer_number == '6':
        df_sensor_info = df_sensor_info[df_sensor_info['SENSOR_OID'].str.slice(10,11) == '6']

    list = df_sensor_info['SENSOR_OID'].tolist()

    return list


def read_csv(file_name):
    df = pd.read_csv(file_name, encoding='euckr')
    df.set_index('TIME_ID', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.index.rename(file_name[file_name.rfind('_') + 1:file_name.rfind('.')], inplace=True)
    return df

# resample 을 하기 전에 오류 값을 제거해야 함
# TEMP,HUMI,PITCH,ROLL,AMBIENT,UV,PRESS,BATTERY,PERIOD,CURRENT,SHOCK,GEOMAG_X,GEOMAG_Y,GEOMAG_Z,VAR_X,VAR_Y,VAR_Z,USN,NTC,UVC
if __name__ == '__main__':


    dir = 'C:/_data/output_db_file_pi'

    list_location = ['고창', '광주', '대구', '대전', '안산']
    list_mounting_position = ['변압기 본체', '부하 개폐기', '완금', '전주', '통신용 함체']
    list_variable = ['TEMP', 'HUMI', 'PITCH', 'ROLL', 'AMBIENT', 'UV', 'PRESS', 'BATTERY', 'PERIOD', 'CURRENT', 'SHOCK', 'GEOMAG_X', 'GEOMAG_Y', 'GEOMAG_Z', 'VAR_X', 'VAR_Y', 'VAR_Z', 'USN', 'NTC', 'UVC']
    list_manufacturer_number = ['1', '4', '6']

    # list_location = ['']
    # list_mounting_position = ['변압기 본체']
    # list_variable = ['PITCH']
    # list_manufacturer_number = ['']


    # pole information read
    df_sensor_info = pd.read_csv('C:/_data/iot_sensor_info.csv', encoding='euckr')
    df_sensor_info = df_sensor_info[df_sensor_info['MOUNTING_POSITION'].notnull()]
    df_sensor_info['FILE_NAME'] = df_sensor_info['SENSOR_OID'].str.slice(10, 11) + '_' + df_sensor_info['POLE_CPTZ_NO'] + '_' + df_sensor_info['SENSOR_OID'] + '.csv'

    for item_location in list_location:
        for item_manufacturer in list_manufacturer_number:
            for item_mounting_position in list_mounting_position:

                list_result = get_list_from_location(df_sensor_info, item_location)
                list_from_mounting_position = get_list_from_mounting_position(df_sensor_info, item_mounting_position)
                list_result = [item for item in list_result if item[item.rfind('_') + 1:item.rfind('.')] in list_from_mounting_position]
                list_from_manufacturer = get_list_from_manufacturer(df_sensor_info, item_manufacturer)
                list_result = [item for item in list_result if item[item.rfind('_') + 1:item.rfind('.')] in list_from_manufacturer]


                list_file_name = []
                cnt = 0
                for path, dirs, files in os.walk(dir):
                    # print(path)
                    for file in files:
                        if file in list_result:
                            cnt += 1
                            # print('FILE_NAME:', file)
                            list_file_name.append(os.path.join(path, file))

                cmt_image = 10
                for idx in range(0, len(list_file_name), cmt_image):
                    list_file_name_part = list_file_name[idx:idx + (cmt_image - 1)]

                    with Pool(processes=16) as pool:
                        list_df = pool.map(read_csv, list_file_name_part)
                        pool.close()
                        pool.join()

                    for item_variable in list_variable:
                        print(item_location + '_' + item_manufacturer + '_' + item_mounting_position + '_' + item_variable + ',', '리스트 길이:', len(list_result))

                        # 아웃풋 디렉토리 체크
                        dir_output = 'C:/_data/output/{}/{}/{}'.format(item_location, item_manufacturer, item_mounting_position)
                        if not os.path.isdir(dir_output):
                            os.makedirs(dir_output)

                        df = pd.DataFrame()
                        df.index.name = 'TIME_ID'

                        bool_plot = False
                        for item in list_df:
                            df_item = pd.DataFrame(item[item_variable])
                            df_item = df_item[df_item[item_variable].notnull()]
                            if len(df_item) > 0:
                                if bool_plot == False:
                                    plt.figure(figsize=(10, 4))
                                df_item = df_item.resample('30min').mean()
                                plt.plot(df_item.index.values, df_item.iloc[:, 0].values, label=df_item.index.name, linewidth=0.7)
                                bool_plot = True

                        if bool_plot == True:
                            legned = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                            plt.title(item_location + '_' + item_manufacturer + '_' + item_mounting_position + '_' + item_variable)
                            plt.xticks(rotation=30)
                            plt.savefig('C:/_data/output/' + item_location + '/' + item_manufacturer + '/' + item_mounting_position + '/' + item_variable + '_' + str(idx + cmt_image) + '.png', bbox_extra_artists=(legned,), bbox_inches='tight')
                            plt.close()
                        else:
                            print(item_location + '_' + item_manufacturer + '_' + item_mounting_position + '_' + item_variable + ',', 'CNT:', idx + cmt_image, ', 데이터가 없습니다.')


# x축 고정
# y축 고정
# y축 제한


