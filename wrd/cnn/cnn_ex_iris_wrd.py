import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style

from wrd.cnn.cnn_ex_iris_perceptron import Perceptron

style.use('seaborn-talk')

matplotlib.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    style.use('seaborn-talk')

    df = pd.read_csv('wrd_clock_pres.csv', dtype=str)
    df.columns = ["state", "datetime", "f_paldang", "f_seongsan", "p_gwangam", "p_oryun", "p_songpa", "p_sinchun", "p_hong", "p_seogang"]

    print(df.values)

    y = df.iloc[0:100, 0]
    X = df.iloc[0:100, [4, 5]]
    print(y)
    print(X)
    y = y.astype(float)
    X = X.astype(float)
    y_val = y.values
    X_val = X.values
    print(X_val[:50, 0])
    print(X_val.dtype)


    plt.scatter(X[:50], X[:50], color = 'r', marker = 'o', label = '1')
    plt.scatter(X[50:100], X[50:100], color='b', marker='x', label='0')
    plt.xlabel('ggotip length(cm)')
    plt.ylabel('ggotbatchim length(cm)')
    plt.legend(loc=4)
    plt.show()

    ppn1 = Perceptron(eta=0.1, n_iter=10000)
    ppn1.fit(X_val, y_val)

    print(ppn1.errors_)