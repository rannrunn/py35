# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import os
from multiprocessing import Pool
from matplotlib import font_manager, rc
import numpy as np
import seaborn as sns
import time

font_name = font_manager.FontProperties(fname="c:\\Windows\\Fonts\\NGULIM.TTF").get_name()
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


def make_plot(dict):
    file_path = dict['file_path']
    pole_id = dict['pole_id']
    plot_type = dict['plot_type']
    resample_how = dict['resample_how']
    item_location = dict['item_location']
    item_manufacturer = dict['item_manufacturer']
    item_mounting_position = dict['item_mounting_position']

    df = pd.read_csv(file_path, encoding='euckr')
    df.set_index('TIME_ID', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.index.rename(file_path[file_path.rfind('_') + 1:file_path.rfind('.')], inplace=True)
    df = df.drop(['PERIOD', 'SHOCK'], axis=1)
    df = df.dropna(axis=1, how='all')
    fig = plt.figure(figsize=(24, 7))
    fig.suptitle(item_location + '_' + item_manufacturer + '_' + item_mounting_position + '_' + pole_id + '_' + df.index.name)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    plt.subplot(131)
    plt.title('\n\n1 Min')
    corr_1_min = df.resample('1Min').mean().corr()
    mask = np.zeros_like(corr_1_min, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr_1_min, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, annot=True, fmt='0.2f')

    plt.subplot(132)
    plt.title('\n\n1 Day')
    corr_1_day = df.resample('1D').mean().corr()
    sns.heatmap(corr_1_day, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, annot=True, fmt='0.2f')

    plt.subplot(133)
    plt.title('\n\n1 Month')
    corr_1_month = df.resample('1M').mean().corr()
    sns.heatmap(corr_1_month, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, annot=True, fmt='0.2f')

    plt.savefig('C:\\_data\\output\\' + plot_type + '\\' + resample_how + '\\' + item_location + '\\' + item_manufacturer + '\\' + item_mounting_position + '\\' + pole_id + '_' + df.index.name + '.png', bbox_inches='tight')
    # plt.show()
    plt.close()
    print(item_location + '_' + item_manufacturer + '_' + item_mounting_position + '_' + pole_id + '_' + df.index.name)


# resample 을 하기 전에 오류 값을 제거해야 함
# TEMP,HUMI,PITCH,ROLL,AMBIENT,UV,PRESS,BATTERY,PERIOD,CURRENT,SHOCK,GEOMAG_X,GEOMAG_Y,GEOMAG_Z,VAR_X,VAR_Y,VAR_Z,USN,NTC,UVC
if __name__ == '__main__':

    start = time.time()

    plot_type = 'correlation'

    dir = 'C:\\_data\\output_db_file_pi'

    list_location = ['고창', '광주', '대구', '대전', '안산']
    list_mounting_position = ['변압기 본체', '부하 개폐기', '완금', '전주', '통신용 함체']
    list_variable = ['TEMP', 'HUMI', 'PITCH', 'ROLL', 'AMBIENT', 'UV', 'PRESS', 'BATTERY', 'PERIOD', 'CURRENT', 'SHOCK', 'GEOMAG_X', 'GEOMAG_Y', 'GEOMAG_Z', 'VAR_X', 'VAR_Y', 'VAR_Z', 'USN', 'NTC', 'UVC']
    list_manufacturer_number = ['1', '4', '6']
    resample_how = '1Min_1Day_1Month'

    # pole information read
    df_sensor_info = pd.read_csv('C:\\_data\\iot_sensor_info.csv', encoding='euckr')
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


                list_file_path = []
                cnt = 0
                for path, dirs, files in os.walk(dir):
                    # print(path)
                    for file in files:
                        if file in list_result:
                            cnt += 1
                            # print('FILE_NAME:', file)
                            list_file_path.append(os.path.join(path, file))


                # 아웃풋 디렉토리 체크
                dir_output = 'C:\\_data\\output\\{}\\{}\\{}\\{}\\{}'.format(plot_type, resample_how, item_location, item_manufacturer, item_mounting_position)
                if not os.path.isdir(dir_output):
                    os.makedirs(dir_output)


                cmt_image = 100
                for idx in range(0, len(list_file_path), cmt_image):
                    list_dict = []
                    for idx_two in range(idx, min(len(list_file_path), idx + cmt_image) , 1):
                        file_name = os.path.basename(list_file_path[idx_two])
                        dict = {}
                        dict['file_path'] = list_file_path[idx_two]
                        if list_file_path[idx_two].find('_') + 1 == list_file_path[idx_two].rfind('_'):
                            dict['pole_id'] = ''
                        else:
                            dict['pole_id'] = file_name[file_name.find('_') + 1:file_name.rfind('_')]
                        dict['plot_type'] = plot_type
                        dict['resample_how'] = resample_how
                        dict['item_location'] = item_location
                        dict['item_manufacturer'] = item_manufacturer
                        dict['item_mounting_position'] = item_mounting_position
                        list_dict.append(dict)

                    with Pool(processes=16) as pool:
                        pool.map(make_plot, list_dict)
                        pool.close()

    print('Total Time:{}'.format(str(round(time.time() - start))))