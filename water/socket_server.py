# coding: utf-8

from _thread import *
import socket

import json
import traceback


def getDictValue(dict, key):
    return dict[key] if key in dict else ''

def on_new_client(conn):
    try:
        print('server : start')
        try:
            data = conn.recv(1024)
        except Exception as e:
            pass
        dec_data = data.decode()
        dict_data = json.loads(dec_data)

        command = getDictValue(dict_data, 'command')
        command_detail = getDictValue(dict_data, 'command_detail')

        if command == 'calculate_statistics':
            if command_detail == 'average':
                dict_data['command_to'] = 'client'
                dict_data['return_value'] = {'average':{'':'', '':'', '':'', '':''}}
                pass
            elif command_detail == 'variance':
                dict_data['command_to'] = 'client'
                dict_data['return_value'] = {'variance':{'':'', '':'', '':'', '':''}}
                pass
            elif command_detail == 'standard_deviation':
                dict_data['command_to'] = 'client'
                dict_data['return_value'] = {'standard_deviation':{'':'', '':'', '':'', '':''}}
                pass
            elif command_detail == 'correlation':
                dict_data['command_to'] = 'client'
                dict_data['return_value'] = {'correlation':{'':'', '':'', '':'', '':''}}
                # 벨리데이션 해야 한다. location_one(locaiton, type), location_two(location, type) 가 모두 있는 지 확인해야 한다.
                pass
        elif command == 'calculate_regression':
            dict_data['command_to'] = 'client'
            dict_data['return_value'] = {{
                "weight_1,weight_2,weight_N,bias" : "15000,10000,9500,1200",
                "multiple_correlation" : "0.88",
                "r_square" : "0.88"
            }}
            pass
        conn.send(json.dumps(dict_data).encode())
        conn.send('end'.encode())
    except Exception as ex:
        conn.send('error'.encode())
        #print('Exception:', ex.value)
        print('server : exception')
        traceback.print_exc()
        print('server : close socket')
        pass

    conn.close()

def client_two(conn):
    pass

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)         # Create a socket object
    s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    host = '' # Get local machine name
    port = 10000                 # Reserve a port for your service.

    s.bind((host, port))        # Bind to the port
    s.listen(5)                 # Now wait for client connection.

    print ('server : Got connection from')

    while True:
       c, addr = s.accept()     # Establish connection with client.
       print ('[*] Connected with ' + addr[0] + ':' + str(addr[1]))
       start_new_thread(on_new_client,(c,))

    s.close()

if __name__ == '__main__':
    main()
