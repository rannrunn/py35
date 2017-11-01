# coding: utf-8
import sys
sys.version
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.nan)
import os
import time
import datetime
from function import statistics
import tensorflow as tf
import traceback

def getData():

    pass


def main(conn):

    try:


        start = time.time()

        option = 'start'  # start, add, predict

        col_names_hongtong_lotte_1 = ['Timestamp','f_hongtong','f_lotte_in','p_3t_1dan','p_3t_incheon_1dan','p_3t_1dan','p_36t','p_3t_1dan','p_oryun_1']
        # 0:Timestamp Indes, 1:Flow Start Index, 2:Flow End Index, 3:Pressure Strat Index, 4:Pressure End Index
        col_names_hongtong_lotte_1_idx = [0,1,2,3,8]
        col_names_hongtong_lotte_1_x_data_cnt = 3
        col_names_hongtong_lotte_2 = ['Timestamp','f_hongtong','f_lotte_in','p_oryun_1','p_songpa_1','p_oryun_2','p_songpa_2']
        col_names_hongtong_lotte_2_idx = [0,1,2,3,6]
        col_names_hongtong_lotte_2_x_data_cnt = 2
        col_names_hongtong_1 = ['Timestamp','f_hongtong','p_songpa_1','p_sincheon_1','p_songpa_1','p_sincheon_2','p_songpa_1','p_lotte_in']
        col_names_hongtong_1_idx = [0,1,1,2,7]
        col_names_hongtong_1_x_data_cnt = 3
        col_names_hongtong_2 = ['Timestamp','f_hongtong','p_sincheon_1','p_hongtong_1','p_sincheon_2','p_hongtong_2']
        col_names_hongtong_2_idx = [0,1,1,2,5]
        col_names_hongtong_2_x_data_cnt = 2

        idx_col = col_names_hongtong_lotte_2_idx
        col_names = col_names_hongtong_lotte_2
        cl_name = '170821_analysis_hongtong_lotte_2'

        idx_time = idx_col[0]
        idx_f_start = idx_col[1]
        idx_f_end = idx_col[2]
        idx_p_start = idx_col[3]
        idx_p_end = idx_col[4]

        col_name = ''
        for i in range(len(col_names)):
            if i == 0:
                col_name = col_names[i]
            else:
                col_name += '_' + col_names[i]



    except Exception as ex:
        print('water regression : train_exception')
        traceback.print_exc()
        pass


