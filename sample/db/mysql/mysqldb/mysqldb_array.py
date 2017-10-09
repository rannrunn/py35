# coding: utf8
import MySQLdb

db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                     user="root",         # your username
                     passwd="1111",  # your password
                     db="water")        # name of the data base

# you must create a Cursor object. It will let
#  you execute all the queries you need
cur = db.cursor()

# Use all the SQL you like
cur.execute("SELECT * FROM TB_WATER")

# 데이타 Fetch
rows = cur.fetchall()
for row in rows:
    print(row)

cur
db.close()