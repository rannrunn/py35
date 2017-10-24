
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import pandas as pd
import csv
import os, sys
#%%

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
            print('the type of colName which I want to find is str')
            for i in range(0, data.shape[1]):
                if (colName[i] == value):
                    tmp.append(i)
            return(tmp)
        elif(type(value) is list):
            print('the type of colName which I want to find is list')
            for j in range(0, len(value)):
                for i in range(0, data.shape[1]):
                    if(value[j] == colName[i]):
                        print(value[j])
                        tmp.append(i)
            return(tmp)
    else:
        print("The elements in two list are not identical")
#%%


# /////////////////////////////////////////////////////////////
#                   read pole_id name from pole_data folder
# /////////////////////////////////////////////////////////////

os.getcwd()
os.chdir("G:/pole_id_data")

#fileNames = os.listdir()
#fileNames_withoutCSV = list(map(lambda x: x.replace(".csv", ""), fileNames))
fileNames_withoutCSV = ['8132X291', '8232P531', '8132X783']
#print(len(fileNames))
#%%
# /////////////////////////////////////////////////////////////
#                   read sensor name
# /////////////////////////////////////////////////////////////

os.getcwd()
os.chdir("G:/bySensor_id_LargePole")
sensorNames = os.listdir()
print(len(sensorNames))
#%%
# /////////////////////////////////////////////////////////////
#                   read sensor data
# /////////////////////////////////////////////////////////////
pole_id_name_tmp = fileNames_withoutCSV[0]
sensor_Index_tmp = [i for i, s in enumerate(sensorNames) if pole_id_name_tmp in s]
print(sensor_Index_tmp)
print(str(sensorNames[2]))

csvName_tmp = str(sensorNames[2])
#print(csvName_tmp)

os.getcwd()
# // read data
start_time = time.time()
data = pd.read_csv(csvName_tmp, encoding='CP949')
print("{0:.2f}".format(time.time() - start_time)) # check merging time

# // remove unnecessary column ('Unnamed: 0', 'sensor_id')
rowNum = data.shape[0]
colNum = data.shape[1]
print("rowNum: " + str(rowNum) + "   " + "colNum: " + str(colNum))

which(data, 'Unnamed: 0') # check whether variable exists
data = data.drop('Unnamed: 0', 1) # remove 'Unnamed: 0'

which(data, 'sensor_id') # check whether variable exists
data = data.drop('sensor_id', 1) # remove 'sensor_id'

data['time_id'] = pd.to_datetime(data['time_id']) # change the type of variable
#%%
# /////////////////////////////////////////////////////////////
#                   merge data according to 'time':
#                   hour/ day/ month
# /////////////////////////////////////////////////////////////
print(type(data))
rowNum = data.shape[0]
colNum = data.shape[1]
print("rowNum: " + str(rowNum) + "   " + "colNum: " + str(colNum))
#data.dtypes
#print(data.iloc[0:3, :])
#print(data.iloc[(rowNum-3):rowNum, :])

# merge data
data = data.fillna(0) # replace Null with 0

#os.getcwd()
#os.chdir("C:\\Users\\soft\\Desktop\\data")
##tmp.to_csv("tmp.csv")


data.set_index(data['time_id'], inplace=True) # set index as timeIndex
data = data.drop('time_id', 1) # delete time column
data.info()
data.head()
data.tail()
#data_

#data_min   = data.resample('1Min', how = {np.sum}) # minute
start_time = time.time()
data_hour  = data.resample('1H', how = {np.mean})   # hour
print("{0:.2f}".format(time.time() - start_time))   # check merging time
#data_day   = data.resample('1D', how = {np.mean})   # day
#data_month = data.resample('1M', how = {np.mean})   # month

# // change colnames

data_hour.columns = list(data)
data_hour.columns
#data_day.columns = list(data)
#data_day.columns
#data_month.columns = list(data)
#data_month.columns