import numpy as np
import torch
import pandas as pd


# df = pd.read_csv('c:/sim_data.csv')
df = pd.read_csv('./sim_data.csv')

list_sw = [5576, 2409, 2410, 28987, 17529, 20245, 18538, 25513, 27775]

np.array(df['current_' + str(5576)].values[:1000]).reshape(10, 100)

for item in list_sw:
    print(df['current_' + str(item)])
    torch.save(np.array(df['current_' + str(item)].values[:1000]).reshape(10, 100), open('./load_data_' + str(item) + '.pt', 'wb'))