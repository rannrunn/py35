# 2차로 받은 폴 데이터의 파일 별 라인수를 세는 코드
# coding: utf-8
import os
import datetime

def main():

    file_names = ['NT_SENSOR_XML.sql']

    for file_name in file_names:

        # 파일 순차 탐색
        cnt = 0
        file_path = 'D:/KEPIOT/KEPIOT/' + file_name

        # 파일이 존재하는 지 체크
        if not os.path.exists(file_path):
            print('not exist:', file_name)
            continue

        # while 문 전에 파일을 읽어야 한다.
        f = open(file_path, 'rt', encoding='UTF8')
        # 파일 내 라인 탐색
        while True:
            line = f.readline()
            # 라인이 존재하는 지 체크
            if not line:
                print(file_path, ': ', cnt)
                # 라인 카운트 개수 초기화
                cnt = 0
                break

            cnt = cnt + 1
            #print(cnt)

if __name__ == '__main__':
    main()


