import MySQLdb


# <<운영 DB접속정보>>
# - DB명 : PIPEDB
# - Port : 3306
# - ID/PW : kpipe / 1234

# <<로컬 DB접속정보>>
# - DB명 : WATER
# - Port : 3306
# - ID/PW : root / 1111


ip = 'localhost'
id = 'root'
password = '1111'
db_name = 'WATER'


def getConnection():
    return MySQLdb.connect(ip, id, password, db_name)