import MySQLdb

ip = 'localhost'
id = 'root'
password = '1111'
db_name = 'WATER'

def getConnection():
    return MySQLdb.connect(ip, id, password, db_name)