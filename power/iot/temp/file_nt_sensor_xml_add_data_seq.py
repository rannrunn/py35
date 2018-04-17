# 2차로 받은 폴 데이터의 파일 별 라인수를 세는 코드
# coding: utf-8
import os
import datetime

def main():

    file_name = 'NT_SENSOR_XML.sql'

    # 파일 순차 탐색
    cnt = 0
    file_path = 'F:/' + file_name
    file_out_path = 'F:/' + 'MODIFY2_' + file_name

    # 파일이 존재하는 지 체크
    if not os.path.exists(file_path):
        print('not exist:', file_name)
        return

    # while 문 전에 파일을 읽어야 한다.
    f = open(file_path, 'rt', encoding='UTF8')
    out = open(file_out_path, 'w', encoding='UTF8')

    lines = ''

    insert_flag = False

    # 파일 내 라인 탐색
    while True:
        line = f.readline()

        # 라인이 존재하는 지 체크
        if not line:
            print(file_path, ': ', cnt)
            # 라인 카운트 개수 초기화
            break

        if line.find('INSERT') != -1:
            line = line.replace('INSERT IGNORE INTO `NT_SENSOR_XML`', 'INSERT IGNORE INTO `NT_SENSOR_XML_MODIFY`')
            line = line.replace('`INS_DT`) VALUES', '`INS_DT`, `DATA_SEQ`) VALUES')
            insert_flag = True
            cnt = 0

        if insert_flag == True:
            line = line.replace('\')','\',\''+ str(cnt) + '\')')
            out.write(line)

        cnt = cnt + 1
        #print(cnt)

if __name__ == '__main__':
    main()


