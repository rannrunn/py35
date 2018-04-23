import pandas as pd
import numpy as np


def indexDayS(da):
    wkd=[pd.to_datetime(da.index[i]).weekday() for i in range(len(da.index))]
    wkd1=np.array([wkd]).T
    da1=pd.DataFrame(wkd1, columns=['dayN'], index=da.index)
    return(da1)


def specificDayS(da1, day=0):
    x=da1.ix[da1.ix[:,'dayN']==day, :len(da1.columns)]
    return(x.index)


x = '2017-10-08'

type(x)

x1 = pd.to_datetime(x)

x1


df = pd.read_csv('kospi.csv')
# print(df)

df.set_index('Date', inplace=True)
df.index = pd.to_datetime(df.index)

kos = df.tail(31)

print(df)

kos1 = indexDayS(kos)

kosday0 = specificDayS(kos1, 0)


print(kosday0.values)





