import numpy as np
import torch
import pandas as pd


# df = pd.read_csv('c:/sim_data.csv')
df = pd.read_csv('./sim_data.csv')

list_sw = [5576, 2409, 2410, 28987, 17529, 20245, 18538, 25513, 27775]

df['current_5576'].to_csv('./5576.csv', header=True)

for item in list_sw:
    pass