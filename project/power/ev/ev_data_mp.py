import pandas as pd
import numpy as np
from multiprocessing import Pool


def func(list_date):

    df = pd.read_csv('C:\\_data\\ev_2016_2017.csv', encoding='euckr')

    df['시작일시'] = pd.to_datetime(df['시작일시'])
    mask = (df['시작일시'] >= list_date[0] + ' 00:00:00') & (df['시작일시'] <= list_date[1] + ' 23:59:59')
    df = df[mask]
    print(df)

    # 전처리 : 충전량이 65를 초과하거나 0인 행 제거
    df = df[df['충전량(kWh)'] <= 65]
    df = df[df['충전량(kWh)'] > 0]

    # 충전시간 추가
    df['충전시간'] = pd.to_datetime(df['시작일시'])
    df['충전시간'] = df['충전시간'] - pd.to_timedelta(df['충전시간'].dt.second, unit='s')

    # 급속 : 0.6
    # 완속 : 0.1

    # 소수점 첫째자리에서 반올림
    df['총충전시간'] = np.nan
    mask = df['충전구분'] == '급속'
    df.loc[mask, '총충전시간'] = round(df['충전량(kWh)'] / 0.6, 0)
    mask = df['충전구분'] == '완속'
    df.loc[mask, '총충전시간'] = round(df['충전량(kWh)'] / 0.1, 0)

    mask = df['총충전시간'] > 0
    df = df[mask]


    df['총충전시간'] = df['총충전시간'].astype(int)

    cnt = 0
    df_result = pd.DataFrame()
    for i in range(len(df)):
        row = df.iloc[i]
        time_delta = None
        row = row.drop(labels=['사업소', '충전방식', '충전량(kWh)', '충전요금(원)', '전기요금(원)', '충전시간(시분)', '시작일시', '종료일시'])
        row['충전시간'] = row['충전시간'] - pd.to_timedelta(1, unit='m')
        for j in range(row['총충전시간']):
            row['충전시간'] = row['충전시간'] + pd.to_timedelta(1, unit='m')
            df_result = df_result.append(row)
        if i % 100 == 0:
            print(list_date[0] + ' : ' + str(i))


    df_result = df_result[['본부', '충전구분', '총충전시간', '충전시간']]
    df_result.to_csv('./ev_hq_charge_amount_' + list_date[0] + '.csv', encoding='euckr')


if __name__ == '__main__':

    list_date = [['2016-01-01', '2016-01-31'], ['2016-02-01', '2016-02-29'],
                 ['2016-03-01', '2016-03-31'], ['2016-04-01', '2016-04-30'],
                 ['2016-05-01', '2016-05-31'], ['2016-06-01', '2016-06-30'],
                 ['2016-07-01', '2016-07-31'], ['2016-08-01', '2016-08-31'],
                 ['2016-09-01', '2016-09-30'], ['2016-10-01', '2016-10-31'],
                 ['2016-11-01', '2016-11-30'], ['2016-12-01', '2016-12-31'],
                 ['2017-01-01', '2017-01-31'], ['2017-02-01', '2017-02-28'],
                 ['2017-03-01', '2017-03-31'], ['2017-04-01', '2017-04-30'],
                 ['2017-05-01', '2017-05-31'], ['2017-06-01', '2017-06-30'],
                 ['2017-07-01', '2017-07-31'], ['2017-08-01', '2017-08-31'],
                 ['2017-09-01', '2017-09-30'], ['2017-10-01', '2017-10-31'],
                 ['2017-11-01', '2017-11-30'], ['2017-12-01', '2017-12-31']
                 ]
    with Pool(processes=24) as pool:
        pool.map(func, list_date)