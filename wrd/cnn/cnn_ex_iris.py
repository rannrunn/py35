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

    df = pd.read_csv('cnn_ex_iris.csv', header=None)

    y = df.iloc[0:100, 4].values
    print(y)
    y = np.where(y=='iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    print(X)

    plt.scatter(X[:50, 0], X[:50, 1], color = 'r', marker = 'o', label = 'setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='b', marker='x', label='versicolor')
    plt.xlabel('ggotip length(cm)')
    plt.ylabel('ggotbatchim length(cm)')
    plt.legend(loc=4)
    plt.show()

    ppn1 = Perceptron(eta=0.1)
    ppn1.fit(X, y)

    print(ppn1.errors_)