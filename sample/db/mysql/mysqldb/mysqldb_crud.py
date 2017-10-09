import MySQLdb

con = MySQLdb.connect('localhost', 'root', '1111', 'WATER')
cur = con.cursor(MySQLdb.cursors.DictCursor)
table = 'TB_TEST'
value_initial_c1 = 1000
value_initial_c2 = 2000
value_update_c1 = 3000

# create table TB_TEST(
#     C1 INT,
#     C2 INT
# )

# SELECT 하는 방법
def select(instruction, table, cur):
    query = 'select * from %s' % (table)
    cur.execute(query);
    results = cur.fetchall()
    print(instruction)
    for row in results:
        print(row)

# INSERT 하는 방법
myDict = {'C1':value_initial_c1,'C2':value_initial_c2}
columns = ','.join(myDict.keys())
placeholders_value = ','.join(['%s'] * len(myDict))
query = 'insert into %s (%s) values (%s)' % (table, columns, placeholders_value)
cur.execute(query, myDict.values())
con.commit()
select('INSERT', table, cur)

# UPDATE 하는 방법
query = 'update %s set %s = %s where %s=%s' % (table, 'c1', value_update_c1, 'c1', value_initial_c1)
cur.execute(query)
con.commit()
select('UPDATE', table, cur)

# DELETE 하는 방법
query = 'delete from %s where %s like %s ' % (table, 'c1', value_update_c1)
cur.execute(query)
con.commit()
select('DELETE', table, cur)