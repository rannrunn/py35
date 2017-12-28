import matplotlib.pyplot as plt
import pandas as pd
import os
import multiprocessing as mp

def getPole(pole_id):
    os.chdir('D:\\dev\\IoT_data\\IoT_233')
    data = pd.read_csv(pole_id + '.csv')
    start = '2016-04-01 00:00:00'
    end = '2017-05-15 23:59:59'

    pole_data = data[['TIME_ID', 'AMBIENT', 'BATTERY', 'HUMI', 'PITCH', 'ROLL', 'PRESS', 'TEMP', 'UV']]
    pole_data['TIME_ID'] = pd.to_datetime(pole_data['TIME_ID'], format='%Y-%m-%d %H:%M:%S')

    # data의 주기가 일정하지 않아 초 단위로 time_id 컬럼을 가진 data_time 생성
    start_time = pd.to_datetime(start, format='%Y-%m-%d %H:%M:%S')
    end_time = pd.to_datetime(end, format='%Y-%m-%d %H:%M:%S')
    time_range = pd.date_range(start_time, end_time, freq='s')
    data_time = pd.DataFrame(time_range)
    data_time.rename(columns={0: 'TIME_ID'}, inplace=True)  # inplace=True: data_time을 직접 변경

    # time_id를 기준으로 data_time과 data_sensor merge
    # data_temp = pd.merge(data_time, pole_data, on='TIME_ID', how='left')
    data_temp=data_time.merge(pole_data, on='TIME_ID', how='left')
    data_temp['TIME_ID'] = pd.to_datetime(data_temp['TIME_ID'], format='%Y-%m-%d %H:%M:%S')
    # index를 time_id로 지정
    data_temp.set_index(data_temp['TIME_ID'], inplace=True)
    data_temp = data_temp.drop('TIME_ID', 1)
    data_temp.index.names = [None]
    data_temp=data_temp.resample('10T').max()


    return data_temp


def saveImage(pole_id):
    print(pole_id)
    df = getPole(pole_id)

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(pole_id)
    ax1 = fig.add_subplot(8, 1, 1)
    ax2 = fig.add_subplot(8, 1, 2)
    ax3 = fig.add_subplot(8, 1, 3)
    ax4 = fig.add_subplot(8, 1, 4)
    ax5 = fig.add_subplot(8, 1, 5)
    ax6 = fig.add_subplot(8, 1, 6)
    ax7 = fig.add_subplot(8, 1, 7)
    ax8 = fig.add_subplot(8, 1, 8)

    ax1.plot(df['AMBIENT'])
    ax2.plot(df['BATTERY'])
    ax3.plot(df['HUMI'])
    ax4.plot(df['TEMP'])
    ax5.plot(df['PITCH'])
    ax6.plot(df['ROLL'])
    ax7.plot(df['UV'])
    ax8.plot(df['PRESS'])

    ax1.set_ylabel('AMBIENT')
    ax2.set_ylabel('BATTERY')
    ax3.set_ylabel('HUMI')
    ax4.set_ylabel('TEMP')
    ax5.set_ylabel('PITCH')
    ax6.set_ylabel('ROLL')
    ax7.set_ylabel('UV')
    ax8.set_ylabel('PRESS')

    plt.grid()
    fig.savefig('D:\\dev\\IoT_data\\parameter8_2\\' + pole_id + '.png', format='png')
    # plt.show()


if __name__ == '__main__':
    os.chdir('D:\\dev\\IoT_data')
    pole = pd.read_csv('pole_id_233.csv')
    for pole_id in pole['pole_id']:
        saveImage(pole_id)

