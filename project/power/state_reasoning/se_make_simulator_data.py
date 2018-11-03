# coding: utf-8
# DL 43 '대평'에 대해 시뮬레이터 연동 작업 진행
# 데이터에 대한 시간별 그래프를 그려 데이터 건전성 시각화
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


dir = 'C:\\_data'

x1 = pd.ExcelFile(os.path.join(dir, '20180710_dl_43.xls'))

df = x1.parse('쿼리')
df.set_index(pd.to_datetime(df['updatetime']), inplace=True)

list_sw = [5576, 2409, 2410, 28987, 17529, 20245, 18538, 25513, 27775]

총합 = 0
평균 = 0
차이 = 0

index_minutes = pd.date_range('2017-11-30 00:00:00', '2018-05-31', freq='4H')
df_result = pd.DataFrame(index=index_minutes)
print('df_result:', df_result)


for item in list_sw:
    df_item = df[df['sw_id'] == item]
    # df_item = df_item.loc['20180101':'20180131']
    df_item[df_item['current_a'] == 0] = np.nan
    df_item[df_item['current_b'] == 0] = np.nan
    df_item[df_item['current_c'] == 0] = np.nan
    df_item['current_' + str(item)] = df_item.loc[:,'current_a'] + df_item.loc[:,'current_b'] + df_item.loc[:,'current_c']
    df_item = df_item.resample('4H').mean()
    # print(df_item)
    df_result['current_' + str(item)] = df_item['current_' + str(item)]

    # 20245 값 수정
    if item == 20245:
        df_result['current_' + str(20245)] = df_result['current_' + str(28987)] - 30
    # print(df_item.loc[:,'current_a'] + df_item.loc[:,'current_b'] + df_item.loc[:,'current_c'])
    # plt.plot(df_result['current_' + str(item)], label=str(item))


print('18538', df_result[df_result.loc[:, 'current_18538'].isnull()]['current_18538'])
print('25513', df_result[df_result.loc[:, 'current_25513'].isnull()]['current_25513'])
print('27775', df_result[df_result.loc[:, 'current_27775'].isnull()]['current_27775'])


df_result.loc[df_result.loc[:, 'current_18538'].isnull(), 'current_18538'] = df_result[df_result.loc[:, 'current_18538'].isnull()]['current_2409'] - 66

# plt.plot(df_result.loc['2017-12-23':'2017-12-28','current_18538'])
plt.plot(df_result['current_18538'], 'r')
plt.plot(df_result['current_25513'], 'b')
plt.plot(df_result['current_27775'], 'g')
plt.legend()
plt.show()
plt.close()


print(df_result[df_result.loc[:, 'current_27775'].isnull()]['current_27775'])

plt.plot(df_result[df_result.loc[:, 'current_18538'].isnull()]['current_18538'], 'r')
plt.plot(df_result['current_25513'], 'b')
plt.plot(df_result['current_27775'], 'g')
plt.legend()
plt.show()
plt.close()

df_result.to_csv('./sim_data.csv')

# 5576 은 중간에 데이터가 들어오지 않는 구간이 있다.
# 2409, 2410 은 같은 데이터가 들어온다.



