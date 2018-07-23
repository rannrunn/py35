import pickle
import pandas as pd

df = pd.read_csv('c:\\sim_data.csv')

df['current_5576']
df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
df['weekday'] = df['Unnamed: 0'].dt.dayofweek
df['label'] = 0

data_section_load = [list(item) for item in zip(df['current_5576'], df['weekday'], df['label'])]

# open a file, where you ant to store the data
file = open('./current_5576.pkl', 'wb')

# dump information to that file
pickle.dump(data_section_load, file)

# close the file
file.close()


with open(str('./current_5576.pkl'), 'rb') as f:
    print(pickle.load(f))