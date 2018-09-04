import os


def get_pole_id_from_path(file_path):
    file_name = os.path.basename(file_path)
    if file_name.find('_') + 1 == file_name.rfind('_'):
        pole_id = ''
    else:
        pole_id = file_name[file_name.find('_') + 1:file_name.rfind('_')]

    return pole_id


def get_sensor_oid_from_path(file_path):
    file_name = os.path.basename(file_path)
    sensor_oid = file_name[file_name.rfind('_') + 1:file_name.rfind('.')]
    return sensor_oid


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

