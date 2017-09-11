import csv

with open('./header.csv', 'w', newline='') as csvfile:
    fieldnames = ['first_name', 'last_name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'first_name' : 'banana', 'last_name' : 'ssang'})
    writer.writerow({'first_name' : 'kong', 'last_name' : 'al'})
    writer.writerow({'first_name' : 'kong', 'last_name' : 'dal'})
