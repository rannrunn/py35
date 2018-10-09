import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

path = './Timeseries.csv'
df = pd.read_csv(path)

# 2016010000 ~

FMT = '%H%M'
df['year'] = df['year'].astype(str)
df['month'] = df['month'].astype(str)
df['month'] = df['month'].apply(lambda x : x.zfill(2))

list_idx = []
for idx_row in range(df.shape[0]):
    tdelta = '0000'
    for idx_col in range(288):
        idx_df = df['year'].values[idx_row] + df['month'].values[idx_row]
        idx_df = idx_df + tdelta
        tdelta = (datetime.datetime.strptime(tdelta, FMT) + datetime.timedelta(minutes=5)).strftime(FMT)
        list_idx.append(idx_df)


df_result = pd.DataFrame(index=[idx for idx in range(len(list_idx))], data=list_idx, columns=['index'])

list_data = []
for idx_row in range(df.shape[0]):
    for idx_col in range(288):
        list_data.append(df.iloc[idx_row, idx_col + 2])

df_result['data'] = list_data

plt.plot(df_result.index, df_result['data'])
plt.show()
plt.close()

df_result.to_csv('./ev.csv')


