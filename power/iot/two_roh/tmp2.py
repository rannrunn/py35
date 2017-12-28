import matplotlib.pyplot as plt
import pandas as pd
import os
import multiprocessing as mp

def getNtc(pole_id):
    os.chdir('D:\\dev\\IoT_data\\IoT')
    # time='2016-09-13'
    data = pd.read_csv(pole_id + '.csv')

    # ntc_data = data[data['time_id'].str.contains(time)]
    ntc_data = data[['time_id', 'ambient']]
    ntc_data['time_id'] = pd.to_datetime(ntc_data['time_id'], format='%Y-%m-%d %H:%M:%S')
    ntc_data.set_index(ntc_data['time_id'], inplace=True)
    ntc_data = ntc_data.drop('time_id', 1)
    ntc_data.index.names = [None]

    return ntc_data

def saveImage(pole_id):
    print(pole_id)
    ntc = getNtc(pole_id)
    fig = plt.figure(figsize=(20, 5))
    fig.suptitle(pole_id)
    ax = fig.add_subplot(111)
    ax.plot(ntc['ambient'])
    plt.grid()
    fig.savefig('D:\\dev\\IoT_data\\ambient\\' + pole_id + '.png', format='png')
    # plt.show()

if __name__=='__main__':

    os.chdir('D:\\dev\\IoT_data')
    pole=pd.read_csv('pole_null.csv')

    # pole_list=pole['pole_id']
    # # print(pole_list)
    # p=mp.Pool(4)
    # p.map(saveImage, pole_list)


    for pole_id in pole['pole_id']:
        saveImage(pole_id)



    # pole='8132Z815'
    # saveImage(pole)
