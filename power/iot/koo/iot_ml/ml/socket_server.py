# coding: utf-8

import datetime
import json
import socket
import traceback
from _thread import *

import validation as valid
import time

import unbalanceLoadInfo as uli
import threading

poleIdArr = [
    '8132X291',
    '8132W782',
    '8232P471',
    '8232R383',
    '8232R152',
    '8132W212',
    '8132W832',
    '8232P531',
    '8132W952',
    '8132Z961',
    '8132X914',
    '8132Q911',
    '8132W231',
    '8132W122',
    '8132X921',
    '8132X152',
    '8132X122',
    '8132W621',
    '8132W981',
    '8132X601'
]

time_month_start = '2017-07-01 00:00:00'
time_month_end = '2017-07-31 23:59:59'

time_day_start = '2017-07-01 00:00:00'
time_day_end = '2017-07-31 23:59:59'

ml = uli.PoleInfo()

# 딕셔너리에서 벨류를 가져오는 함수
def getDictValue(dict, key):
    return dict[key] if key in dict else ''

# 새로운 소켓 링크 생성
def on_new_client(conn, client, port):

    for item in threading.enumerate():
        print(item)

    try:
        data = conn.recv(1024)
        dec_data = data.decode()
        dict_data = json.loads(dec_data)

        print ('[*] ' + client + ':' + port + ' : Request JSON Data : ', dict_data)

        command = getDictValue(dict_data, 'command')
        command_detail = getDictValue(dict_data, 'command_detail')

        dict_data['command_to'] = 'client'
        dict_data['return_value'] = ''

        if dict_data['command'] == 'day':
            for pole_id in poleIdArr:
                ml.getDailyInfo(pole_id, time_day_start, time_day_end)
        elif dict_data['command'] == 'month':
            for pole_id in poleIdArr:
                ml.getDailyInfo(pole_id, time_month_start, time_month_end)
                break

    except Exception as ex:
        traceback.print_exc()
    finally:
        response_data = json.dumps(dict_data)
        conn.send(response_data.encode())
        conn.close()

    print ('[*] ' + client + ':' + port + ' : Response JSON Data : ', response_data)
    print ('[*] ' + client + ':' + port + ' : Connection End ')


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)         # Create a socket object
    s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    host = '' # Get local machine name
    port = 12580                 # Reserve a port for your service.

    s.bind((host, port))        # Bind to the port
    s.listen(5)                 # Now wait for client connection.

    print ('Server start :', datetime.datetime.now())

    while True:
        c, addr = s.accept()     # Establish connection with client.
        client = addr[0]
        port = str(addr[1])
        print ('[*] ' + client + ':' + port + ' : Connection Start')
        start_new_thread(on_new_client,(c, client, port))

    s.close()

if __name__ == '__main__':
    main()
