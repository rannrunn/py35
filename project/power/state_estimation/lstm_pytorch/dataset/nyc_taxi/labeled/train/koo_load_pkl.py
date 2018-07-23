import pickle
import pandas as pd


with open(str('./nyc_taxi.pkl'), 'rb') as f:
    pkl = pickle.load(f)
    print(type(pkl))
    print(len(pkl))
    print(pkl)
