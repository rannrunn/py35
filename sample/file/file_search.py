# coding: utf-8
import os

for root, dirs, files in os.walk("C:\\_data\\부하데이터"):
    for file in files:
        if file.endswith(".xls"):
            print(os.path.join(root, file))
