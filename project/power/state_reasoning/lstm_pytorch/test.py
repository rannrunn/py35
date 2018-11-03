import pickle
import pandas as pd

df = pd.read_csv('c:\\sim_data.csv')

df['current_5576']
df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
df['minute'] = df['Unnamed: 0'].dt.hour * 60
df['weekday'] = df['Unnamed: 0'].dt.dayofweek
df['label'] = 0

list_data = [list(item) for item in zip(df['current_5576'], df['minute'], df['weekday'], df['label'])]


