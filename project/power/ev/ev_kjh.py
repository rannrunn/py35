import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./Timeseries.csv')

print(df)

list_df = []
df_result = pd.DataFrame(columns=['year', 'month', 'charge', 'split_num'])
for idx in range(1, 289, 1):
    df_temp = df[['year', 'month', 'x' + str(idx)]]
    df_temp['year'] = df_temp['year'].astype(str)
    df_temp['month'] = df_temp['month'].astype(str)
    df_temp['month'] = df_temp['month'].apply(lambda x: str(x).zfill(2))
    df_temp['yyyymmdd'] = df_temp['year'] + df_temp['month']
    df_temp['split_num'] = idx
    df_temp.columns = ['year', 'month', 'charge', 'yyyymmdd','split_num']
    list_df.append(df_temp)
    print('end:', idx)

df_result = pd.concat(list_df)

plt.plot(df_temp['charge'])
plt.show()
plt.close()