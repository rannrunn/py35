# coding: utf-8
import socket
import time
import json

def main():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 12580))

    data = {}
    # abnormal_judgment , learning_start, learning_add, predict_discharge
    data['command'] = 'learning_start[]'

    json_data = json.dumps(data)

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