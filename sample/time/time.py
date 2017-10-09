import datetime

now = datetime.datetime.now()
print (now)

print (now + datetime.timedelta(hours=1, minutes=23, seconds=10))



s1 = '10:33:26'
s2 = '11:15:49' # for example
FMT = '%H:%M:%S'
tdelta = datetime.datetime.strptime(s2, FMT) - datetime.datetime.strptime(s1, FMT)


s1 = '2017-06-20'
s2 = '2017-06-19'
s3 = '0000-00-01'
FMT = '%Y-%m-%d'
tdelta = (datetime.datetime.strptime(s1, FMT) + datetime.timedelta(days=1)).strftime(FMT)
print(tdelta)
