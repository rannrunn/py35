# coding: utf-8

import datetime
import json
import socket
import traceback
from _thread import *

import calculate_regression
import calculate_statistics
import validation as valid

# 딕셔너리에서 벨류를 가져오는 함수
def getDictValue(dict, key):
    return dict[key] if key in dict else ''

# 새로운 소켓 링크 생성
def on_new_client(conn, client, port):

    try:
        data = conn.recv(1024)
        dec_data = data.decode()
        dict_data = json.loads(dec_data)

        print ('[*] ' + client + ':' + port + ' : Request JSON Data : ', dict_data)

        command = getDictValue(dict_data, 'command')
        command_detail = getDictValue(dict_data, 'command_detail')

        dict_data['command_to'] = 'client'
        dict_data['return_value'] = ''

        # dictionary 데이터를 벨리데이션 하여 오류가 있을 경우 리턴
        dict_data = valid.validate_json(dict_data)
        if 'error' in dict_data and dict_data['error'] != '':
            raise Exception

        if command == 'calculate_statistics':
            if command_detail == 'average':
                dict_data = calculate_statistics.calculate(dict_data)
            elif command_detail == 'variance':
                dict_data = calculate_statistics.calculate(dict_data)
            elif command_detail == 'standard_deviation':
                dict_data = calculate_statistics.calculate(dict_data)
            elif command_detail == 'correlation':
                dict_data = calculate_statistics.calculate(dict_data)
        elif command == 'calculate_regression':
            dict_data = calculate_regression.calculate(dict_data)

    except Exception as ex:
        #traceback.print_exc()
        if dict_data['error'] == '':
            dict_data['error'] = 'socket server error'
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
