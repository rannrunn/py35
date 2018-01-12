import pandas as pd
import os

cnt = 0
df = pd.DataFrame(columns=['POLE_ID', 'SENSOR_ID', 'PART_NAME', 'CNT', 'PERIOD', 'VARIABLE'])
for path, dir, files in os.walk('C:/csv/'):
    for filename in files:
        file, ext = os.path.splitext(filename)
        cnt += 1
        if ext == '.csv':
            print("%s%s" % (path, filename))
            df_read = pd.read_csv('{}{}'.format(path, filename), encoding='euc-kr')
            period = file[9:10]
            variable = file[19:].upper()
            df_read['PERIOD'] = period
            df_read['VARIABLE'] = variable
            print(period)
            print(variable)

        if variable =='NTC':
            print(df_read)
        print(cnt)
        print(df.columns)
        df = pd.concat([df, df_read], axis=0)

df = df.sort_values(by=['VARIABLE', 'PERIOD', 'POLE_ID', 'SENSOR_ID', 'PART_NAME'], axis=0)

df['INDEX'] = [idx for idx in range(len(df.index.values))]
df = df.set_index('INDEX')

df.to_csv('C:/csv/total.csv')




