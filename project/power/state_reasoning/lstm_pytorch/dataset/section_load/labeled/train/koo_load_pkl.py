import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


with open(str('./section_load.pkl'), 'rb') as f:
    pkl = pickle.load(f)
    print(type(pkl))
    print(len(pkl))
    print(pkl)

arr = np.array(pkl)


plt.plot(arr.T[0])
plt.show()

