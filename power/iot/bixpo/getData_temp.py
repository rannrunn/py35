# /////////////////////////////////////////////////////////////
#                                 library
# /////////////////////////////////////////////////////////////

import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import pandas as pd
import os, sys
import multiprocessing as mp

# /////////////////////////////////////////////////////////////
#                                 user-defined function
# /////////////////////////////////////////////////////////////



def which(data, value):
    tmp = []
    colName = list(data)
    # check all elements in two lists are identical
    # // if(set(columns_Order) == set(list(data))):
    if(isinstance(value, str)):
        value_tmp = [value]
    elif(isinstance(value, list)):
        value_tmp = value
    if(set(list(data)).intersection(set(value_tmp)) == set(list(value_tmp))):
        if(type(value) is str):
            # print('the type of colName which I want to find is str')
            for i in range(0, data.shape[1]):
                if (colName[i] == value):
                    tmp.append(i)
            return(tmp)
        elif(type(value) is list):
            # print('the type of colName which I want to find is list')
            for j in range(0, len(value)):
                for i in range(0, data.shape[1]):
                    if(value[j] == colName[i]):
                        # print(value[j])
                        tmp.append(i)
            return(tmp)
    else:
        print("The elements in two list are not identical")

def setDataPath(abs_path):
        os.chdir(abs_path)
# /////////////////////////////////////////////////////////////
#                   get pole_id list
# /////////////////////////////////////////////////////////////

# setDataPath("c:/getData")
# pole_id_list = pd.read_csv("pole_id.csv")
# pole_id_list = list(pole_id_list['pole_id'])



# // output: pandas dataframe

# // directory (pole data)
setDataPath("D:\\dev\\py_data\\pole_id_data2")




def getData(pole_id, partName_Index, variableName, period, startTime=0, endTime=0):
    # /////////////////////////////////////////////////////////////
    #                                 read data
    # /////////////////////////////////////////////////////////////
    start_time = time.time()
    tmp_fileName = pole_id + ".csv"
    print(tmp_fileName)
    data = pd.read_csv(tmp_fileName, encoding='utf-8')
    data.rename(columns={'Unnamed: 0': 'id'}, inplace=True) # colname 변경
    columns_Order = ['sensor_id', 'time_id', 'temp', 'humi', 'ambient',
                    'battery', 'current', 'press', 'shock', 'uv', 'pitch', 'roll']

    colIndex = which(data, columns_Order)
    data = data.iloc[:, colIndex]
    # sort data by 'time_id'
    data = data.sort_values('time_id')
    data = data.reset_index(drop=True)

    # /////////////////////////////////////////////////////////////
    #                                 Check % of null elements
    # /////////////////////////////////////////////////////////////

    # count null by each columns
    tmp = pd.isnull(data).sum()/data.shape[0]
    tmp = pd.DataFrame(tmp)
    tmp = tmp.iloc[:, 0].apply(lambda x: float("{0:.2f}".format(x)))
    tmp = pd.DataFrame(tmp)
    tmp = tmp.iloc[:, 0].apply(lambda x: x*100)
    tmp = pd.DataFrame(tmp)

    # get names of variable whose null % is lower than 30%
    tmp_Index = []
    for i in range(0,tmp.shape[0]):
        if(tmp.iloc[i,0] <= 100):
            tmp_Index.append(i)

    columns_Selected = list(tmp.iloc[tmp_Index, :].index)

    # /////////////////////////////////////////////////////////////
    #                Select variables with row % null value
    # /////////////////////////////////////////////////////////////

    # select specific columns (which have low null %)
    colIndex = which(data, columns_Selected)
    data = data.iloc[:, colIndex]

    data.reset_index(drop=True) # 행번호 다시 메김

    # parameter: select variable
    if isinstance(variableName, list):
        variableName_original = variableName[:]
        if not 'pitch' in variableName_original:
            variableName_original.append('pitch')
        if not 'roll' in variableName_original:
            variableName_original.append('roll')
        if not 'shock' in variableName_original:
            variableName_original.append('shock')
        if not 'battery' in variableName_original:
            variableName_original.append('battery')
    elif isinstance(variableName, str):
        variableName_original = [variableName, 'pitch', 'roll', 'shock', 'battery']

    if isinstance(variableName, list):
        variableName.append('time_id')
        variableName.append('sensor_id')
        if not 'pitch' in variableName:
            variableName.append('pitch')
        if not 'roll' in variableName:
            variableName.append('roll')
        if not 'shock' in variableName:
            variableName.append('shock')
        if not 'battery' in variableName:
            variableName.append('battery')
    elif isinstance(variableName, str):
        variableName = [variableName, 'pitch', 'roll', 'shock', 'battery']
    # variableName_original = variableName
    # variableName = [variableName, 'time_id', 'sensor_id']
    colIndex = which(data, variableName)
    data = data.iloc[:, colIndex]

    # /////////////////////////////////////////////////////////////
    #                    divide data according to partName
    # /////////////////////////////////////////////////////////////
    sensor_id_unique = list(data.sensor_id.unique())
    sensor_id_Index_tmp = partName_Index
    print(partName_Index)
    print(sensor_id_unique[partName_Index])
    data_withOneSensor_tmp = data.loc[data['sensor_id'] == sensor_id_unique[sensor_id_Index_tmp], :] #
    data_withOneSensor_tmp['time_id'] = pd.to_datetime(data_withOneSensor_tmp['time_id']) # change type of time variable


    # /////////////////////////////////////////////////////////////
    #                                 addressing time index
    # /////////////////////////////////////////////////////////////
    # change str to datetime
    data_withOneSensor_tmp['time_id'] = pd.to_datetime(data_withOneSensor_tmp['time_id'])

    # parameter: find range of time
    if((startTime is 0) & (endTime is 0)):
        time_id_range = [min(data_withOneSensor_tmp['time_id']), max(data_withOneSensor_tmp['time_id'])]
    elif(isinstance(startTime, str) & isinstance(endTime, str)):
        startTime = pd.to_datetime(startTime)
        endTime = pd.to_datetime(endTime)
        time_id_range = [startTime, endTime]

    # discard first, end day (which doesn`t have full hours)
    time_id_range[0] = pd.to_datetime(time_id_range[0]) + datetime.timedelta(days=1)
    time_id_range[0] = time_id_range[0].replace(hour=0, minute=0, second = 0)
    time_id_range[1] = pd.to_datetime(time_id_range[1]) - datetime.timedelta(days=1)
    time_id_range[1] = time_id_range[1].replace(hour=0, minute=0, second = 0)

    # make time index
    time_index = pd.date_range(time_id_range[0], time_id_range[1], freq='s')
    time_index = pd.DataFrame(time_index)

    # rename column names
    time_index.rename(columns={0: 'time_id'}, inplace=True)

    # join the two dataframes using pd.merge
    data = pd.merge(time_index, data_withOneSensor_tmp, on='time_id', how='left')

    # /////////////////////////////////////////////////////////////
    #                   merge data according to 'time':
    #                   hour/ day/ month
    # /////////////////////////////////////////////////////////////
    data.set_index(data['time_id'], inplace=True)  # set index as timeIndex
    data = data.drop('time_id', 1)  # delete time column

    # aggregate data
    # parameter: period
    data_aggregated = data.resample(period, how = {np.mean}) # aggregate according to time

    # // change colnames
    data_aggregated.columns = [variableName_original]

    print("{0:.2f}".format((time.time() - start_time)/60) + ' min')  # check merging time
    return(data_aggregated)



def saveimage(filename):

    # /////////////////////////////////////////////////////////////
    #                   function parameters (example)
    # /////////////////////////////////////////////////////////////
    # // input: parameters
    # pole_id = pole_id_list[0]
    # partName_Index = 0
    # variableName = ['temp'] # 'temp' # 'temp' or ['temp', 'humi']
    # period = '1D' # 1Min/ 1H/ 1D/ 1M
    # startTime = 0 # default = 0: first date
    # endTime = 0   # default = 0: last  date

    pole_id = filename.split('.')[0]

    partName_Index = 0
    variableName = ['temp']  # 'temp' # 'temp' or ['temp', 'humi']
    period = '1D'  # 1Min/ 1H/ 1D/ 1M
    startTime = 0  # default = 0: first date
    endTime = 0  # default = 0: last  date
    # # // code example
    tt1 = getData(pole_id = pole_id,
                 partName_Index = partName_Index,
                 variableName = variableName,
                 period = period,
                 startTime = startTime,
                 endTime = endTime)

    partName_Index = 1
    variableName = ['temp']  # 'temp' # 'temp' or ['temp', 'humi']
    period = '1D'  # 1Min/ 1H/ 1D/ 1M
    startTime = 0  # default = 0: first date
    endTime = 0  # default = 0: last  date
    tt2 = getData(pole_id=pole_id,
                 partName_Index=partName_Index,
                 variableName=variableName,
                 period=period,
                 startTime=startTime,
                 endTime=endTime)

    partName_Index = 2
    variableName = ['temp']  # 'temp' # 'temp' or ['temp', 'humi']
    period = '1D'  # 1Min/ 1H/ 1D/ 1M
    startTime = 0  # default = 0: first date
    endTime = 0  # default = 0: last  date
    tt3 = getData(pole_id=pole_id,
                 partName_Index=partName_Index,
                 variableName=variableName,
                 period=period,
                 startTime=startTime,
                 endTime=endTime)

    tt_all=[tt1, tt2, tt3]
    tt4=pd.concat(tt_all)

    # type(tt)
    dstart = datetime.datetime(2016,4,1)
    dend = datetime.datetime(2017,5,31)

    fig = plt.figure(figsize=(15, 13))
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)
    fig.suptitle(pole_id)
    ax1.plot(tt1['temp'])
    ax2.plot(tt2['temp'])
    ax3.plot(tt3['temp'])
    ax4.plot(tt4['temp'])

    ax1.set_title()
    ax1.set_ylabel('TEMP')
    ax1.set_xlim(dstart, dend)
    ax1.set_ylim( [-25,60] )

    ax2.set_ylabel('TEMP')
    ax2.set_xlim(dstart, dend)
    ax2.set_ylim( [-25,60] )

    ax3.set_ylabel('TEMP')
    ax3.set_xlim(dstart, dend)
    ax3.set_ylim( [-25,60] )

    ax4.set_ylabel('TEMP')
    ax4.set_xlim(dstart, dend)
    ax4.set_ylim([-25, 60])

    fig.savefig('D:\\dev\\py_data\\result\\' + pole_id + '.png', format='png')

    # fig = plt.figure(figsize=(15, 10))
    # ax1 = fig.add_subplot(6, 1, 1)
    # ax2 = fig.add_subplot(6, 1, 2)
    # ax3 = fig.add_subplot(6, 1, 3)
    # ax4 = fig.add_subplot(6, 1, 4)
    # ax5 = fig.add_subplot(6, 1, 5)
    # ax6 = fig.add_subplot(6, 1, 6)
    # fig.suptitle(pole_id)
    # ax1.plot(tt['temp'])
    # ax2.plot(tt['humi'])
    # ax3.plot(tt['pitch'])
    # ax4.plot(tt['roll'])
    # ax5.plot(tt['shock'])
    # ax6.plot(tt['battery'])
    # ax1.set_ylabel('TEMP')
    # ax1.set_xlim(dstart, dend)
    # ax1.set_ylim( [-25,60] )
    # ax1.get_xlabel('0')
    # ax2.set_ylabel('HUMI')
    # ax2.set_xlim(dstart, dend)
    # ax2.set_ylim( [-10,110] )
    # ax3.set_ylabel('PITCH')
    # ax3.set_xlim(dstart, dend)
    # ax3.set_ylim( [-100,100] )
    # ax4.set_ylabel('ROLL')
    # ax4.set_xlim(dstart, dend)
    # ax4.set_ylim( [-100,100] )
    # ax5.set_ylabel('SHOCK')
    # ax5.set_xlim(dstart, dend)
    # ax5.set_ylim( [-1.1,1.1] )
    # ax6.set_ylabel('BATTERY')
    # ax6.set_xlim(dstart, dend)
    # ax6.set_ylim( [-10,110] )
    # fig.savefig('D:\\dev\\py_data\\result\\pole_6_100\\' + pole_id + '.png', format='png')

if __name__ == '__main__':
    filenames=['8232P471']

    # filenames = ['8232P471',
    #              '8132G005',
    #              '8132X914',
    #              '8132W952',
    #              '8132X152',
    #              '8232P531',
    #              '8132X921',
    #              '8232P142',
    #              '8232P371',
    #              '8132X783',
    #              '8232P345',
    #              '8132W981',
    #              '8132X601',
    #              '8132X391',
    #              '8132X171',
    #              '8132X291',
    #              '8132X761',
    #              '8132Q911',
    #              '8132W122',
    #              '8132W782',
    #              '8232R152',
    #              '8132Z961',
    #              '8132Z763',
    #              '8232R383',
    #              '8232P472',
    #              '8132X402',
    #              '8232P544',
    #              '8132X361',
    #              '8132X362',
    #              '8132G101',
    #              '8132X506',
    #              '8132X464',
    #              '8132X815',
    #              '8232P571',
    #              '8132W162',
    #              '8132X831',
    #              '8232P042',
    #              '8132X916',
    #              '8132Z814',
    #              '8132Q903',
    #              '8232R121',
    #              '8232R372',
    #              '8232R242',
    #              '8232R432',
    #              '8232P542',
    #              '8132D824',
    #              '8132G511',
    #              '8132G512',
    #              '8132X681',
    #              '8132X272']


    # for filename in os.listdir('C:/pole_id_data_all'):
    #     filenames.append(filename)

    print(filenames)

    p = mp.Pool(4)
    p.map(saveimage, filenames)




