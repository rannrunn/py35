import pickle
import pandas as pd

df = pd.read_csv('c:\\sim_data.csv')

df['current_5576']
df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
df['minute'] = df['Unnamed: 0'].dt.hour * 60
df['weekday'] = df['Unnamed: 0'].dt.dayofweek
df['label'] = 0

data_section_load = [list(item) for item in zip(df['current_5576'], df['minute'], df['weekday'], df['label'])]
data_section_load_test = data_section_load[:800]
data_section_load_train = data_section_load[800:]

# open a file, where you ant to store the data
file = open('./section_load.pkl', 'wb')
# dump information to that file
pickle.dump(data_section_load_test, file)


# open a file, where you ant to store the data
file = open('./section_load.pkl', 'wb')
# dump information to that file
pickle.dump(data_section_load_train, file)


# close the file
file.close()

