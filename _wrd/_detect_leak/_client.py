# coding: utf-8
import socket
import os
import time

def main():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 12580))

    data = 'client_test'

    client.send(data.encode())

    while True:
        response = client.recv(4096)
        print (response.decode())
        if response.decode() == 'end':
            break

if __name__ == "__main__":
    start = time.time()
    main()
    print('communication time: %f' % (time.time() - start))