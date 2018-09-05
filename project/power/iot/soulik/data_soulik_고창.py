import pandas as pd
import matplotlib
import os
import time

cnt = 0
cnt_line = 0

f = open('C:\\고창.csv', 'r')
while True:
    cnt += 1
    line = f.readline()
    if not line: break
    list_line = line.split(',')
    if len(list_line) != cnt_line:
        cnt_line = len(list_line)
        print('cnt:', cnt,  ', length:', len(list_line) - 1, ', contents:', list_line)

f.close()


# 고창
# 1행 11라인
# 7556행 13라인 -> 7557행 14라인
# 12061행 12라인 -> 12062행 13라인
# 13671행 8라인 -> 13672행 9라인
# 17197행 11라인
# 18093행 14라인
# 19037행 13라인
# 20016행 9라인


def write(list, oid):
    f1 = open('C:\\_data\\고창_' + oid + '.csv', 'w')
    for item in list:
        f1.write(item)
    f1.close()
    # print(list)
    print('write:', 'C:\\_data\\고창_' + oid + '.csv')

# 고창
list = [1, 7556, 12061, 13671, 17197, 18093, 19037, 20016]
list_line_all = []
list_line_data = []
cnt = 0
cnt_line = 0

f = open('C:\\고창.csv', 'r')
oid = ''
while True:
    cnt += 1
    line = f.readline()
    if not line:
        print('break')
        write(list_line_data, list_line_all[-1][1])
        break
    list_line = line.split(',')
    oid = list_line[1]
    if cnt in list[1:]:
        write(list_line_data, list_line_all[-1][1])
        list_line_data = []
        list_line_all = []
        cnt_line = len(list_line)
    list_line_data.append(line)
    list_line_all.append(list_line)

f.close()




df = pd.DataFrame( columns=['time', 'oid', 'temp', 'lux', 'uvc', 'ntc', 'pitch', 'roll', 'var_x', 'var_y', 'var_z', 'geomag_x', 'geomag_y', 'geomag_z', 'usn', 'battery'])
# df.set_index(pd.to_datetime(df['time']), inplace=True)
# df.index
cnt = 0
cnt_all = 0
for path, dirs, filenames in os.walk('C:\\_data\\IoT_고창'):
    print(filenames)
    for filename in filenames:
        print(filename)
        if filename.find('.bak') > -1:
            continue

        filepath = os.path.join(path, filename)
        df_temp = pd.read_csv(filepath)
        print(df_temp)
        # df_temp.set_index(pd.to_datetime(df['time']), inplace=True)
        df = pd.concat([df, df_temp], axis=0)

df = df[['time', 'oid', 'temp', 'lux', 'uvc', 'ntc', 'pitch', 'roll', 'var_x', 'var_y', 'var_z', 'geomag_x', 'geomag_y', 'geomag_z', 'usn', 'battery']]
df.set_index(pd.to_datetime(df['time']), inplace=True)
df.sort_index(ascending=True, inplace=True)

df.to_csv('C:\\_data\\고창_통합.csv', index=False)


print('전체 라인:', cnt_all)


