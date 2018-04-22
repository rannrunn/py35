# coding: utf-8
import time

import pandas as pd

if __name__ == '__main__':

    start_time_main = time.time()

    time_start = '2016-04-01 00:00:00'
    time_end = '2017-05-23 23:59:59'
    resample_how = '10T'

    path_dir = 'G:\\'
    file_name = '1차폴의센서별10분당데이터유무.csv'

    df_left = pd.DataFrame()
    df_right = pd.DataFrame()

    for idx in range(540):
        cnt = idx + 1
        if cnt % 20 == 0 or cnt == 540:
            df_right = pd.read_csv(path_dir + str(cnt) + '_' + file_name, encoding = "euc-kr")
            df_right.set_index('Unnamed: 0', inplace=True)
            df_left = pd.merge(df_left, df_right, left_index=True, right_index=True, how='outer')

    print(df_left)

    df_left.to_csv(path_dir + file_name)

    print('Total Time:' + str(round(time.time() - start_time_main, 4)))




