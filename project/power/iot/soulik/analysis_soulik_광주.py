import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df_temp = pd.read_csv('C:\\_data\\IoT\\광주_통합.csv')
df_temp['time'] = pd.to_datetime(df_temp['time'])
df_temp.set_index(['time', 'oid'], inplace=True)
df_temp.sort_index(ascending=True, inplace=True)


for item in df_temp.columns:
    if item in ['time', 'oid'] or item.find('Unnamed') > -1:
        continue

    print(item)
    if df_temp[item].dtypes != 'float':
        df_temp[item] = df_temp[item].apply(lambda x:np.nan if x.replace('.', '').isdigit() == False else x)
        df_temp[item] = df_temp[item].astype(float)




for item in df_temp.loc[df_temp['oid'].duplicated() == False, 'oid']:
    for var in df_temp.columns:
        if var in ['time', 'oid'] or var.find('Unnamed') > -1:
            continue

        plt.plot(df_temp.loc[df_temp['oid'] == item].index, df_temp.loc[df_temp['oid'] == item, var], label=var)


plt.title(item)
plt.xticks(rotation=30)
plt.legend()
plt.show()
plt.close()




