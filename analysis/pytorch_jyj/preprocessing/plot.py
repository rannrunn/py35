# https://stackoverflow.com/questions/9008370/python-2d-contour-plot-from-3-lists-x-y-and-rho

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # 3d plot
from sklearn import decomposition   # PCA
import preprocessing as pre
import numpy as np
import pandas as pd


import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import pylab


def contour_2d(x_, y_, z_):
    if isinstance(x_, list):
        x_ = np.array(x_)
    if isinstance(y_, list):
        y_ = np.array(y_)
    if isinstance(z_, list):
        z_ = np.array(z_)

    _xi, _yi = np.linspace(x_.min(), x_.max(), 100), np.linspace(y_.min(), y_.max(), 100)
    _xi, _yi = np.meshgrid(_xi, _yi)

    # Interpolate
    _rbf = scipy.interpolate.Rbf(x_, y_, z_, function='linear')     # this takes long time
    _zi = _rbf(_xi, _yi)

    plt.imshow(_zi, vmin=z_.min(), vmax=z_.max(), origin='lower',
               extent=[x_.min(), x_.max(), y_.min(), y_.max()])
    plt.scatter(x_, y_, c=z_)
    plt.colorbar()
    plt.show()


def contour_3d(x_, y_, z_):
    if isinstance(x_, list):
        x_ = np.array(x_)
    if isinstance(y_, list):
        y_ = np.array(y_)
    if isinstance(z_, list):
        z_ = np.array(z_)

    _xi, _yi = np.linspace(x_.min(), x_.max(), 100), np.linspace(y_.min(), y_.max(), 100)
    _xi, _yi = np.meshgrid(_xi, _yi)

    # Interpolate
    _rbf = scipy.interpolate.Rbf(x_, y_, z_, function='linear')     # this takes long time
    _zi = _rbf(_xi, _yi)

    _fig = pylab.figure()
    _ax = Axes3D(_fig)
    _ax.plot_surface(_xi, _yi, _zi, cmap="autumn_r", lw=0, rstride=1, cstride=1)
    plt.show()


def do_pca(x_, n_col=3):
    _pca = decomposition.PCA(n_components=n_col)
    _pca.fit(x_)
    _x_pca = _pca.transform(x_)
    return _x_pca, _pca


def plot_3d(x_, y_):
    fig = plt.figure(1, figsize=(5, 5))
    plt.clf()  # clear figure
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()

    if not isinstance(x_, pd.DataFrame):
        x_ = pd.DataFrame(x_)

    ax.scatter(x_.ix[:, 0], x_.ix[:, 1], x_.ix[:, 2], c=y_, cmap=plt.cm.spectral,
               edgecolor='k')


if __name__ == "__main__":
    data = pre.get_iris()
    x, y = pre.split_x_y(data, col="Species")

    x_pca, pca = do_pca(x_=x, n_col=3)

    plot_3d(x, y)
    plot_3d(x_pca, y)






