import matplotlib.pyplot as plt
import pandas as pd
import os
import multiprocessing as mp

def getTemp(pole_id):
    sensor = '변압기 본체'
    os.chdir('D:\\dev\\IoT_data\\IoT')

    data = pd.read_csv(pole_id + '.csv')

    # temp_data = data[data['time_id'].str.contains(time)]
    temp_data = data.loc[data['part_name'] == sensor, ['time_id', 'temp']]
    temp_data['time_id'] = pd.to_datetime(temp_data['time_id'], format='%Y-%m-%d %H:%M:%S')
    temp_data.set_index(temp_data['time_id'], inplace=True)
    temp_data = temp_data.drop('time_id', 1)
    temp_data.index.names = [None]

    return temp_data

def saveImage(pole_id):
    print(pole_id)
    temp = getTemp(pole_id)
    fig = plt.figure(figsize=(20, 5))
    fig.suptitle(pole_id)
    ax = fig.add_subplot(111)
    ax.plot(temp['temp'][1000:1500], '.')
    plt.grid()
    fig.savefig('D:\\dev\\IoT_data\\변압기개수\\' + pole_id + '.png', format='png')

if __name__=='__main__':

    os.chdir('D:\\dev\\IoT_data')
    pole=pd.read_csv('pole_null.csv')
    pole_list=pole['pole_id']
    # print(pole_list)
    p=mp.Pool(4)
    p.map(saveImage, pole_list)

