# 소켓 클라이언트를 구동하는 소스
# coding: utf-8

import socket
import time
import json

def main():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 12580))

    # calculate_statistics
    # calculate_regression
    command = 'calculate_regression'
    # average, variance, standard_deviation, correlation
    command_detail = 'correlation'

    dict = {}
    dict['command'] = command
    dict['command_to'] = 'server'
    time_start = '2017-07-01 00:00:00'
    time_end = '2017-11-21 00:00:00'
    dict['input'] = 'PDILCGS.PDI_FI_TOTAL.F_CV'
    dict['output'] = 'PDILCGS.PDI_PT601C.F_CV'

    if command == 'calculate_statistics':
        if command_detail == 'correlation':
            dict['command_detail'] = command_detail
            dict['sector'] = '001'
            dict['table'] = 'RDR01MI_TB'
            dict['time_start'] = time_start
            dict['time_end'] = time_end
        else:
            dict['command_detail'] = command_detail
            dict['sector'] = '001'
            dict['table'] = 'RDR01MI_TB'
            dict['time_start'] = time_start
            dict['time_end'] = time_end
    elif command == 'calculate_regression':
        dict['sector'] = '001'
        dict['table'] = 'RDR01MI_TB'
        dict['time_start'] = time_start
        dict['time_end'] = time_end

    # abnormal_judgment , learning_start, learning_add, predict_discharge

    json_data = json.dumps(dict)

    print('request_data : ', json_data)

    client.send(json_data.encode())

    try:
        response = client.recv(4096)
        dec_data = response.decode()
        print('response_data :', dec_data)
    except Exception as ex:
        # print('Exception:')
        print('Exception')

if __name__ == '__main__':
    start = time.time()
    main()
    print('communication time: %f' % (time.time() - start))