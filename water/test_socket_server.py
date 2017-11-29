# coding: utf-8

from _thread import *
import socket

import json
import datetime

def getDictValue(dict, key):
    return dict[key] if key in dict else ''

def on_new_client(conn, client, port):
    response_data = ''
    try:
        data = conn.recv(1024)
        dec_data = data.decode()
        dict_data = json.loads(dec_data)

        print ('[*] ' + client + ':' + port + ' : Request JSON Data : ', dict_data)

        command = getDictValue(dict_data, 'command')
        command_detail = getDictValue(dict_data, 'command_detail')

        dict_data['command_to'] = 'client'

        if command == 'calculate_statistics':
            if command_detail == 'average':
                dict_data['return_value'] = {'average':{'':'', '':'', '':'', '':''}}
            elif command_detail == 'variance':
                dict_data['return_value'] = {'variance':{'':'', '':'', '':'', '':''}}
            elif command_detail == 'standard_deviation':
                dict_data['return_value'] = {'standard_deviation':{'':'', '':'', '':'', '':''}}
            elif command_detail == 'correlation':
                dict_data['return_value'] = {'correlation':{'':'', '':'', '':'', '':''}}
                # 벨리데이션 해야 한다. location_one(locaiton, type), location_two(location, type) 가 모두 있는 지 확인해야 한다.
        elif command == 'calculate_regression':
            dict_data['return_value'] = {
                "weight_1,weight_2,weight_3,bias" : "15000,10000,9500,1200",
                "multiple_correlation" : "0.88",
                "r_square" : "0.50"
            }

        response_data = json.dumps(dict_data)
        conn.send(response_data.encode())
    except Exception as ex:
        response_data = 'Exception'
        conn.send(response_data.encode())

    print ('[*] ' + client + ':' + port + ' : Response JSON Data : ', response_data)
    print ('[*] ' + client + ':' + port + ' : Connection End ')

    conn.close()

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
