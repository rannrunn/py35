import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

df = pd.read_csv('D:\\IoT\\output_db_file_pi\\1_9691H162_1.2.481.1.1.2.58030141967349.csv')
df.set_index('TIME_ID', inplace=True)
df.index = pd.to_datetime(df.index)

print(df)

plt.plot(df.index.values, df['TEMP'], label=df.index.name, linewidth=0.7)
plt.show()

# df = df[df['TEMP'] != 300]
# df = df[df['TEMP'] != 0]



df = df.resample('3min').mean()
df_shift_1 = df.shift(-1)
df_shift_2 = df.shift(-2)
df_shift_3 = df.shift(-3)


df = df.drop(df[((df['TEMP'].isnull()) & (df_shift_1['TEMP'].notnull())) | ((df['TEMP'].isnull()) & (df_shift_1['TEMP'].isnull()) & (df_shift_2['TEMP'].notnull())) | ((df['TEMP'].isnull()) & (df_shift_1['TEMP'].isnull()) & (df_shift_2['TEMP'].isnull()) & (df_shift_3['TEMP'].notnull())) ].index)

# df.to_csv('C:\\_data\\xx.csv')

plt.plot(df.index.values, df['TEMP'], label=df.index.name, linewidth=0.7)
plt.show()


df_ori = copy.deepcopy(df)


df['DIFF'] = df['TEMP'].diff()

df['UP'] = df['DIFF'] > 10
df['DOWN'] = df['DIFF'] < -10


df.loc[df['UP'], 'TEMP'] = np.nan
df.loc[df['DOWN'], 'TEMP'] = np.nan


# df.to_csv('C:\\_data\\xx_last.csv')


# print(df['UP'])
plt.plot(df.index.values, df_ori['TEMP'], label=df.index.name, linewidth=0.7)
plt.show()

# 10분 동안 5도를 초과하여 상승하는 경우
plt.plot(df.index.values, df_ori['TEMP'], label=df.index.name, linewidth=0.7)
plt.plot(df.index.values, df['TEMP'], label=df.index.name, linewidth=0.7)
plt.show()


