import os
import datetime

file_first = "G:/MariaDB/kepco_iot/data/sensor_"
file_date = '2017-06-20'
last_date_add_one = "2017-09-21"
FMT = '%Y-%m-%d'

# 파일 순차 탐색
while True:
    cnt = 0
    file_name = file_first + file_date
    # 파일이 존재하는 지 체크

    if file_date == last_date_add_one:
        break
    #print(file_name)

    file_date = (datetime.datetime.strptime(file_date, FMT) + datetime.timedelta(days=1)).strftime(FMT)
    # False 와 비교하여도 동작한다.
    if not  os.path.exists(file_name):
        print('not exist:', file_name)
        continue

    # while 문 전에 파일을 읽어야 한다.
    f = open(file_name, 'r')
    # 파일 내 라인 탐색
    while True:
        line = f.readline()
        # 라인이 존재하는 지 체크
        if not line:
            print(file_name, ': ', cnt)
            # 라인 카운트 개수 초기화
            cnt = 0
            break

        cnt = cnt + 1
        #print(cnt)





