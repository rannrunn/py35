import pandas as pd
import matplotlib
import os
import time

cnt = 0
cnt_line = 0

f = open('C:\\광주.csv', 'r')
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
    f1 = open('C:\\_data\\광주_' + oid + '.csv', 'w')
    for item in list:
        f1.write(item)
    f1.close()
    # print(list)
    print('write:', 'C:\\_data\\광주_' + oid + '.csv')

# 광주
list = [1, 4861, 7342, 7750]
list_line_all = []
list_line_data = []
cnt = 0
cnt_line = 0

f = open('C:\\광주.csv', 'r')
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






cnt = 0
cnt_all = 0
for path, dirs, filenames in os.walk('C:\\_data\\IoT'):
    print(filenames)
    for filename in filenames:

        if filename.find('.bak') > -1:
            continue

        filepath = os.path.join(path, filename)
        cnt = 0
        f = open(filepath, 'r')
        while True:
            line = f.readline()
            if not line:
                print('filename:', filename, ', cnt:', cnt)
                cnt_all += cnt
                break
            cnt += 1
        df = pd.read_csv(filepath)
        df = df[df.duplicated(['time'], keep=False)]

        print(df[df.iloc[:, 0] == 'oid'])


print('전체 라인:', cnt_all)


