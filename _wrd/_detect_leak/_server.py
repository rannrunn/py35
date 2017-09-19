# coding: utf-8
#!/usr/bin/python
# This is server.py file

from _thread import *
import socket
import server_test as so
import wrd_tf_regression_dynamic_x_two as wrd
import json
import traceback

def on_new_client(conn):
    try:
        while True:
            print("server start")
            try:
                data = conn.recv(1024)
            except Exception as e:
                pass
            dec_data = data.decode()
            dict_data = json.loads(dec_data)

            cmd = dict_data["command"]

            if cmd == "learning_start":
                #result = so.test(dec_data, conn)
                wrd.main(conn)
                print(cmd)
            elif cmd == "learning_add":
                print(cmd)
            elif cmd == "predict":
                print(cmd)
            else:
                print("cmd nothing")

            result = so.test(dec_data, conn)



        conn.send("end".encode())
    except Exception as ex:
        #print("Exception:", ex.value)
        #conn.send("error".encode())
        print("server exception:")
        traceback.print_exc()
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
