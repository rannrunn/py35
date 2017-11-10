# coding: utf-8
import socket
import time
import json

def main():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 10000))


    # calculate_statistics_abnormal
    # calculate_abnormal
    # calculate_statistics
    # calculate_regression
    command = 'calculate_statistics'
    # average, variance, standard_deviation,
    command_detail = 'correlation'

    dict = {}
    dict['command'] = command
    dict['command_to'] = 'server'
    time_start = '2017-07-01 00:00:00'
    time_end = '2017-08-01 00:00:00'

    if command == 'calculate_statistics':
        if command_detail == 'correlation':
            dict['command_detail'] = command_detail
            dict['sector'] = ''
            dict['table'] = 'RDR01MI_TB'
            dict['time_start'] = time_start
            dict['time_end'] = time_end
        else:
            dict['command_detail'] = command_detail
            dict['sector'] = ''
            dict['table'] = 'RDR01MI_TB'
            dict['time_start'] = time_start
            dict['time_end'] = time_end
    elif command == 'calculate_regression':
        dict['sector'] = ''
        dict['table'] = 'RDR01MI_TB'
        dict['time_start'] = time_start
        dict['time_end'] = time_end

    print(dict)

    # abnormal_judgment , learning_start, learning_add, predict_discharge

    json_data = json.dumps(dict)

    print('client : JSON_print: ', json_data)

    client.send(json_data.encode())

    while True:
        print('client : start')
        try:
            response = client.recv(4096)
            print ('client : response :', response.decode())
            dec_data = response.decode()
            if dec_data == 'end':
                print('client : dec_data : end')
                break
            elif dec_data == 'error':
                print('client : dec_data : error')
                break
            else:
                print('clien : dict_data :', dec_data)
            print('client : while continue')
        except Exception as ex:
            # print('Exception:')
            print('client : Exception')
            pass

    print('client : .........')

if __name__ == '__main__':
    start = time.time()
    main()
    print('client : communication time: %f' % (time.time() - start))