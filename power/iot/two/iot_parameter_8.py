import matplotlib.pyplot as plt
import pandas as pd
import traceback
import time
import os

variable_all = ['TIME_ID', 'AMBIENT', 'BATTERY', 'HUMI', 'TEMP', 'PITCH', 'ROLL', 'UV', 'PRESS']
variable_plot = ['AMBIENT', 'BATTERY', 'HUMI', 'TEMP', 'PITCH', 'ROLL', 'UV', 'PRESS']
limit_data = {'AMBIENT':[0, 81900],'BATTERY':[0, 100],'HUMI':[0, 100],'TEMP':[-15, 65],'PITCH':[-180, 180],'ROLL':[-90, 90],'UV':[0, 20.48],'PRESS':[300, 1100]}
limit_ylim = {'AMBIENT':[-100, 82000], 'BATTERY':[-5, 105], 'HUMI':[-5, 105], 'TEMP':[-15, 65], 'PITCH':[-185, 185], 'ROLL':[-95, 95], 'UV':[-1, 22], 'PRESS':[-10, 1110]}

def getPole(dir_data, pole_id, time_start, time_end, resample_how):
    data = pd.read_csv(dir_data + pole_id + '.csv')

    # 1. 깊은복사를 해야 경고 메시지가 뜨지 않는다.
    # 2. 깊은복사를 하면 데이터 처리 속도가 빨라진다.
    pole_data = data[variable_all].copy()
    # inplace True => 제자리에서 변경(새로운 DataFrame을 생성하지 않음)
    pole_data.set_index('TIME_ID', inplace=True)
    pole_data.index = pd.to_datetime(pole_data.index)

    # 아웃라이어 제거
    for item in variable_plot:
        if item in limit_data.keys():
            mask = pole_data[item] < limit_data[item][0]
            pole_data.loc[mask, item] = None

            mask = pole_data[item] > limit_data[item][1]
            pole_data.loc[mask, item] = None

    # resample을 하기 위해서는 인덱스의 형식이 datetime 형식이어야 한다.
    pole_data = pole_data.resample(resample_how).mean()
    date_range = pd.date_range(time_start, time_end, freq=resample_how)
    pole_data = pole_data.reindex(date_range)

    return pole_data

def saveImage(path_dir, pole_id, time_start, time_end, resample_how):
    start_time = time.time()

    # 데이터 디렉토리 체크
    dir_data = path_dir + 'data\\'
    if not os.path.isdir(dir_data):
        print('데이터 디렉토리가 존재하지 않습니다.')
        return

    # 아웃풋 디렉토리 체크
    dir_output = path_dir + 'output\\'
    if not os.path.isdir(dir_output):
        os.mkdir(dir_output)

    # 전주의 데이터를 가져온다.
    df = getPole(dir_data, pole_id, time_start, time_end, resample_how)

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(pole_id)

    # plot 할 때 모든 인덱스를 출력하기 위해 맨 처음 행과 맨 마지막 행에 값 삽입
    # 꼼수이기 때문에 다른 방법이 있다면 바꿔야 한다.
    df.iloc[0, :] = 0
    df.iloc[-1, :] = 0

    cnt = 0
    for item in variable_plot:
        cnt = cnt + 1
        exec('ax' + str(cnt) + ' = fig.add_subplot(' + str(len(variable_plot)) + ', 1, ' + str(cnt) + ')')
        eval('ax' + str(cnt) + '.plot(df[\'' + item + '\'])')
        eval('ax' + str(cnt) + '.set_ylabel(\'' + item + '\')')
        eval('ax' + str(cnt) + '.set_ylim(' + str(limit_ylim[item]) + ')')

    print(pole_id + ':' + str(round(time.time() - start_time, 4)))

    try:
        fig.savefig(dir_output + pole_id + '.png', format='png')
        plt.close(fig)
    except Exception as ex:
        traceback.print_exc()
        print('Exception:' + pole_id)
    finally:
        plt.close(fig)

if __name__ == '__main__':
    start_time_main = time.time()
    path_dir = 'F:\\IOT\\'
    pole = pd.read_csv(path_dir + 'pole_id_233.csv')
    time_start = '2016-04-01 00:00:00'
    time_end = '2017-05-15 23:59:59'
    resample_how = '10T'

    for pole_id in pole['pole_id']:
        saveImage(path_dir, pole_id, time_start, time_end, resample_how)

    print('Total Time:' + str(round(time.time() - start_time_main, 4)))


