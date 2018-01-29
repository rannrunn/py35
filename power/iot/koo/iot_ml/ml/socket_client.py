# coding: utf-8

import socket
import time
import json
import traceback

def main():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 12580))

    # calculate_statistics
    # calculate_regression
    command = 'day'
    # average, variance, standard_deviation, correlation
    command_detail = 'average'

    dict = {}
    dict['command'] = command
    dict['command_detail'] = ''

    # abnormal_judgment , learning_start, learning_add, predict_discharge

    json_data = json.dumps(dict)

    print('request_data : ', json_data)

    client.send(json_data.encode())

    try:
        response = client.recv(4096)
        dec_data = response.decode()
        dec_data_json = json.loads(dec_data)
        print('response_data :', dec_data)
        print('return_value :', dec_data_json['return_value'])
    except Exception as ex:
        traceback.print_exc()
        print('Exception')

if __name__ == '__main__':
    start = time.time()
    main()
    print('communication time: %f' % (time.time() - start))