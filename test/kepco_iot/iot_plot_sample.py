import matplotlib.pyplot as plt
import MySQLdb

con = MySQLdb.connect('localhost', 'root', '1111', 'kepco')
cur = con.cursor(MySQLdb.cursors.DictCursor)

list_time_id = []
list_battery = []

def select():
    query = "\nselect time_id, battery from tb_iot_pole" \
            + "\nwhere pole_id = 'Test004' and sensor_id = '1.2.481.1.6.1.0005F2'" \
            + "\norder by time_id asc" \
            + "\nlimit 100;"
    cur.execute(query);
    results = cur.fetchall()
    for row in results:
        try:
            float(row['battery'])
            list_battery.append(float(row['battery']))
        except Exception as e:
            pass

select()

plt.plot (list_battery)

plt.show()