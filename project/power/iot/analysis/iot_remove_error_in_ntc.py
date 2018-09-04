import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool
import math
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:\\Windows\\Fonts\\NGULIM.TTF").get_name()
rc('font', family=font_name)

def plot_data(file_name):

    if file_name[:2] != '6_':
        print('파일명:' + file_name + ', 2차 디바이스가 아니기 때문에 종료합니다.')
        return

    fig = plt.figure(figsize=(10,6))

    dir = 'C:\\_data'

    sensor_oid = file_name[file_name.rfind('_') + 1:file_name.rfind('.')]

    df = pd.read_csv(os.path.join(dir + '\\output_db_file_pi', file_name))
    df['TIME_ID'] = pd.to_datetime(df['TIME_ID'])
    df['TIME'] = df['TIME_ID']
    df.set_index('TIME_ID', inplace=True)

    # 유중온도가 None 인 행은 모두 삭제
    df = df.drop(df[df['NTC'].isnull()].index)

    if len(df) == 0:
        print('파일명:' + file_name + ', 유중온도가 존재하지 않아 종료합니다.')
        return

    if len((df['TIME'].diff().dt.total_seconds() / 10).mode()) > 0:
        df['TIME_DIFF'] = df['TIME'].diff()
        time_diff = (round(df['TIME_DIFF'].dt.total_seconds() / 30.00)).mode()[0] * 30 * (1/2)
        resample_how = str(time_diff) + 'S'
        print('resample_how:', resample_how, 'time_diff:', time_diff)
    else:
        print('주기를 구할 수 없어 종료합니다.')
        return
    # 계산한 주기 이내에 데이터가 있는 경우 후에 전송된 데이터 삭제
    df.loc[df['TIME'].diff().dt.total_seconds() < time_diff, 'LESS_THAN_PERIOD'] = True
    df_dence_data = df[df['LESS_THAN_PERIOD'] == True]
    df_drop_dense_data = df.drop(df_dence_data.index)

    print('df_len:', len(df), 'df_drop_dense_data_len:', len(df_drop_dense_data), 'df_dence_data_len:', len(df_dence_data))

    plt.subplot(211)
    plt.title('\n')
    plt.plot(df.index, df['NTC'], 'o', label='원본 데이터', markersize=1)
    plt.plot(df_dence_data.index, df_dence_data['NTC'], 'ro', label='밀집 데이터', markersize=3)
    plt.legend()

    # 수식에 의해 차분 임계치 계산하여 초과할 경우 드롭 처리
    # 주기 180초일 때 기본 수식 : 6 * logx + 8
    df_drop_dense_data.loc[:, 'NTC_DIFF'] = df_drop_dense_data.loc[:, 'NTC'].diff()
    df_drop_dense_data.loc[((df_drop_dense_data['NTC_DIFF'].abs() > (9 * np.log10(df_drop_dense_data['TIME_DIFF'].dt.total_seconds() / 180) + 16))), 'DROP'] = True
    mask_drop_1 = df_drop_dense_data['DROP'] == True
    mask_drop_1_shift = mask_drop_1.shift(-1)
    mask_drop_1_shift[-1] = False
    df_drop_error_data = df_drop_dense_data.drop(df_drop_dense_data[mask_drop_1 | mask_drop_1_shift].index)
    # 최대, 최소 범위를 벗어날 경우 드롭 처리
    mask_drop_2 = (df_drop_error_data['NTC'] > 125) | (df_drop_error_data['NTC'] < -40)
    df_drop_error_data = df_drop_error_data.drop(df_drop_error_data[mask_drop_2].index)

    df_result_drop = df_drop_error_data.resample(resample_how).mean()
    df_result_error_data = df_drop_dense_data.drop(df_drop_error_data.index).resample(resample_how).mean()



    fig.suptitle(file_name.replace('.csv', '') + ', RESAMPLE_HOW:' + resample_how + ', ori_len:' + str(len(df)) + ', result_len:' + str(len(df_drop_error_data)))
    plt.subplot(212)
    # 비정상(or 이상) 데이터가 제거된 데이터
    plt.plot(df_result_drop.index.values, df_result_drop['NTC'], 'o', label='비정상, 이상, 밀집 데이터가 제거된 데이터', markersize=1)
    # 비정상 데이터
    plt.plot(df_result_error_data.index.values, df_result_error_data['NTC'], 'ro', label='비정상, 이상 데이터', markersize=3)
    plt.legend()

    plt.savefig(os.path.join(dir + '\\output\\plot_remove_error_in_ntc', file_name.replace('.csv', '') + '.png'), bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':

    dir = 'C:\\_data\\output_db_file_pi'

    if not os.path.isdir('C:\\_data\\output\\plot_remove_error_in_ntc'):
        os.makedirs('C:\\_data\\output\\plot_remove_error_in_ntc')

    list_files = []
    for path, dirs, files in os.walk(dir):
        list_files = files
        break

    print(list_files)
    with Pool(processes=16) as pool:
        pool.map(plot_data, list_files)
        pool.close()


