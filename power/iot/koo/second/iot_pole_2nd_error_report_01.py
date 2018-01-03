# coding: utf-8
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

date_range = pd.date_range('2017-06-20 00:00:00', '2017-12-28 00:00:00', freq='1D')
df_one = pd.DataFrame()
df_one = df_one.reindex(date_range)

date_range_part = pd.date_range('2017-12-13', '2017-12-28', freq='1D')
list_part_data = [str(item)[0:10] for item in date_range_part.values]


df_one['FLAG'] = 1
zero_file_size = []
pwd = "D:\\010_data\\kepco\\iot_pole\\2nd\\data"
for path, dirs, files in os.walk(pwd):
    for file in files:
        file_stat = os.stat(os.path.join(path, file))
        if file_stat.st_size == 0:
            zero_file_size.append(file[7:])


# 정상데이터
for i in range(len(df_one.index)):
    for j in zero_file_size:
        if str(df_one.index[i])[:10] == j :
            df_one['FLAG'][i] = 0
    for j in list_part_data:
        if str(df_one.index[i])[:10] == j :
            df_one['FLAG'][i] = 0


# 없는데이터
df_two = df_one.copy()
df_two['FLAG'] = 0
for i in range(len(df_two.index)):
    for j in zero_file_size:
        if str(df_two.index[i])[:10] == j :
            df_two['FLAG'][i] = 1


# 일부데이터
df_three = df_one.copy()
df_three['FLAG'] = 0
for i in range(len(df_three.index)):
    for j in list_part_data:
        if str(df_three.index[i])[:10] == j :
            df_three['FLAG'][i] = 1


fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(111)
# ax.bar 함수는 x값과 y값을 따로 넣어야 한다. df 변수를 그대로 넣어서는 plot 되지 않는다.
ax.bar(df_one.index.values, df_one['FLAG'], width=1, color='blue')
ax.bar(df_one.index.values, df_two['FLAG'], width=1, color='red')
ax.bar(df_one.index.values, df_three['FLAG'], width=1, color='yellow')
ax.set_ylim([0, 3])
ax.set_xlabel('기   간', fontsize=20)
ax.xaxis_date()

plt.xticks(fontsize=20, rotation=20)

blue_patch = mpatches.Patch(color='blue', label='정상데이터')
red_patch = mpatches.Patch(color='red', label='데이터없음')
yellow_patch = mpatches.Patch(color='yellow', label='일부데이터')
plt.legend(handles=[blue_patch, red_patch, yellow_patch], fontsize="x-large")

plt.show()
plt.close()




