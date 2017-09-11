# coding: utf-8
def test(input, conn):
    print("11")
    conn.send("xxxx1".encode())
    print("22")
    conn.send("xxxx2".encode())
    return "end"