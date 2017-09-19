# coding: utf-8
import socket
import time
import json

def main():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 12580))

    data = {}
    data["command"] = 'learning_start'

    json_data = json.dumps(data)

    print('JSON_print: ', json_data)

    client.send(json_data.encode())

    while True:
        print("client start")
        try:
            response = client.recv(4096)
            print("xx")
            print (response.decode())
            dec_data = response.decode()
            if dec_data == 'end':
                print("111111111")
                break
            elif dec_data == "error":
                print("222222222")
                break
            print("00000000")
        except Exception as ex:
            # print("Exception:")
            print('errorrrrrrrrr')
            pass

    print(".........")

if __name__ == "__main__":
    start = time.time()
    main()
    print('communication time: %f' % (time.time() - start))