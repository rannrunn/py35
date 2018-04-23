import os

path_dir = 'C:/dat/'

file_list = os.listdir(path_dir)

for item in file_list:

    file_name = item
    file_path = path_dir + file_name

    if os.path.exists(file_path):
        f = open(file_path, 'r')
        while True:
            line = f.readline()
            if not line:
                print('종료')
                break