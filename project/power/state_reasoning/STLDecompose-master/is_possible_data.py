import copy
import os
from read_data import read_data as rd
from save_model import save_model as sm
from pre_processing import preprocessing_LSTM as pre
import numpy as np


def get_is_possible_data_dict(path, file_name, data_dict, version):
    # 정상/비정상 check
    try:
        is_possible_data_dict = sm.load_pkl(path.to_save_data, file_name.is_possible_dict + "_ver_" + version + ".pkl")

    except IOError:
        is_possible_data_dict = check_possible(data_dict)
        sm.save_pkl(path.to_save_data, is_possible_data_dict, "is_possible_data_dict", version)

    return is_possible_data_dict


def is_possible(data, col_name='load'):
    # todo: code 정리
    # --- 현빈 ver --- #
    if col_name not in data.columns:
        data.rename(columns={data.columns[0]: 'time', data.columns[1]: col_name}, inplace=True)
        data = data[['time', col_name]]

    # todo: abnormal rate 왜 이렇게 높게 나오지 ... ?

    num_nan = len(data[col_name][np.isnan(data[col_name])])
    num_successive_same_val = len(data[col_name][data[col_name] > 0]) - len(data[col_name][data[col_name] > 0].unique())
    num_under_0 = len(data[data[col_name] <= 0])
    num_of_abnormal_data =  num_nan + num_successive_same_val + num_under_0
    rate_of_abnormal_data = num_of_abnormal_data / len(data[col_name])

    # ///
    # pre.remove_abnormal_from_pd_series(data)
    # ///
    if rate_of_abnormal_data >= 0.3:    # 원래 0.3
        is_possible_data = False
    else:

        recent_data = data[col_name][-6*24*7:]        # data point 1008 (6*24*7) equals to 7days
        num_of_abnormal_data = len(recent_data[np.isnan(recent_data)]) + len(recent_data[recent_data > 0]) \
                               - len(recent_data[recent_data > 0].unique()) + len(recent_data[recent_data <= 0])
        recent_abnormal_rate = num_of_abnormal_data/len(recent_data)

        if recent_abnormal_rate >= 0.5:
            is_possible_data = False
        else:
            is_possible_data = True

    return is_possible_data

# def is_possible(data, col_name='load'):
    # --- 영재 ver --- #
    # n_row = data.shape[0]
    #
    # # --- preprocess: 1) Remove outlier --- #
    # outliers = []
    # for idx in range(n_row):
    #     if data['load'][idx] <= 0:
    #         outliers.append(idx)
    #
    #     elif (idx < n_row - 2) and (data['load'][idx] == data['load'][idx + 1]):
    #         outliers.append(idx + 1)
    #
    # abnormal_rate = (len(outliers) / n_row)
    #
    # if abnormal_rate > 0.3:
    #     return False
    # else:
    #     return True
    #
    # return is_possible_data


def make_empty_dict():
    """
    head_office_dict[head_office][sub_station][dl_id]
    """
    path = r"D:\소프트팩토리\소프트팩토리_대전\data\PQMS\부하데이터"
    os.chdir(path)
    os.getcwd()

    # --- 본부(head office) 폴더 --- #
    head_office = rd.get_sub_files(path)

    # make empty head office dict
    head_office_dict = {}

    # --- 변전소(sub-station) 폴더 --- #
    for h_o in head_office:
        path_substation = path + "\\" + h_o
        sub_stations = rd.get_sub_files(path_substation)

        sub_stations_dict = {}
        for s_s in sub_stations:
            sub_stations_dict[s_s] = None

        head_office_dict[h_o] = sub_stations_dict

    # --- D/L id 폴더 --- #
    for h_o in head_office:
        path_substation = path + "\\" + h_o
        sub_stations = rd.get_sub_files(path_substation)

        for s_s in sub_stations:
            path_dl_id_folder = path_substation + "\\" + s_s
            DL_id_folders = rd.get_sub_files(path_dl_id_folder)

            DL_id_folders_dict = {}
            for dl_id in DL_id_folders:
                DL_id_folders_dict[dl_id] = None

            head_office_dict[h_o][s_s] = DL_id_folders_dict

    return head_office_dict


def check_possible(data_dict):
    is_possible_data_dict = copy.deepcopy(data_dict)
    head_office = list(data_dict.keys())
    for h_o in head_office:
        sub_station = list(data_dict[h_o].keys())
        for s_s in sub_station:
            dl_ids = list(data_dict[h_o][s_s].keys())
            for dl_id in dl_ids:

                data = data_dict[h_o][s_s][dl_id]
                if is_possible(data):
                    is_possible_data_dict[h_o][s_s][dl_id] = True   # True
                else:
                    is_possible_data_dict[h_o][s_s][dl_id] = False  # False

    return is_possible_data_dict
