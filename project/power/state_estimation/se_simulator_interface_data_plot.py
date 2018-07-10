# DL 43 '대평'에 대해 시뮬레이터 연동 작업 진행
# 데이터에 대한 시간별 그래프를 그려 데이터 건전성 시각화
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


dir = 'C:\\_data'

x1 = pd.ExcelFile(os.path.join(dir, '20180710_dl_43.xls'))


print(x1.parse('쿼리'))

df = x1.parse('쿼리')
df.set_index(pd.to_datetime(df['updatetime']), inplace=True)

list_sw = [5576, 2409, 2410, 28987, 17529, 20245, 18538, 25513, 27775]

for idx in range(len(list_sw)):
    df_item = df[df['sw_id'] == list_sw[idx]]
    df_item = df_item.loc['20180101':'20180131']
    plt.plot(df_item['current_a'] + df_item['current_b'] + df_item['current_c'], label=str(list_sw[idx]))
plt.legend()
plt.show()
plt.close()
