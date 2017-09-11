# coding: utf-8
#!/usr/bin/python
# This is server.py file

from _thread import *
import socket
import os
import server_test as so

def on_new_client(conn):
    try:
        while True:
            data = conn.recv(1024)
            dec_data = data.decode()
            print(dec_data)
            print(os.path.dirname(dec_data))

            result = so.test(dec_data, conn)

            conn.send(result.encode())
    except Exception as ex:
        print('close socket')
        pass

    conn.close()

def client_two(conn):
    pass

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)         # Create a socket object
    s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    host = '' # Get local machine name
    port = 12580                 # Reserve a port for your service.

    s.bind((host, port))        # Bind to the port
    s.listen(5)                 # Now wait for client connection.

    print ('Got connection from')

    while True:
       c, addr = s.accept()     # Establish connection with client.
       print ('[*] Connected with ' + addr[0] + ':' + str(addr[1]))
       start_new_thread(on_new_client,(c,))

    s.close()

if __name__ == "__main__":
    main()
