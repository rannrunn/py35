import MySQLdb

ip = 'localhost'
id = 'kpipe'
password = '1234'
db_name = 'PIPEDB'


def getConnection():
    return MySQLdb.connect(ip, id, password, db_name)