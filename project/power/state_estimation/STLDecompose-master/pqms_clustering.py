import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.cluster import KMeans


X = np.array([[7, 5], [5, 7], [7, 7], [4, 4], [4, 6], [1, 4],
              [0, 0], [2, 2], [8, 7], [6, 8], [5, 5], [3, 7]])
plt.scatter(X[:, 0], X[:, 1], s=100)
plt.show()


model1 = KMeans(n_clusters=2, init="random", n_init=1,
                max_iter=1, random_state=1).fit(X)


c0, c1 = model1.cluster_centers_
c0, c1


model1.score(X)


def plot_cluster(model, c0, c1):
    plt.scatter(X[model.labels_ == 0, 0],
                X[model.labels_ == 0, 1], s=100, marker='v', c='r')
    plt.scatter(X[model.labels_ == 1, 0],
                X[model.labels_ == 1, 1], s=100, marker='^', c='b')
    plt.scatter(c0[0], c0[1], s=200, c="r")
    plt.scatter(c1[0], c1[1], s=200, c="b")
    plt.show()


def kmeans_df(model, c0, c1):
    df = pd.DataFrame(np.hstack([X,
                                 np.linalg.norm(X - c0, axis=1)[:, np.newaxis],
                                 np.linalg.norm(X - c1, axis=1)[:, np.newaxis],
                                 model.labels_[:, np.newaxis]]),
                      columns=["x0", "x1", "d0", "d1", "c"])
    return df

def calc_new_centroid(model):
    c0_new = (X[model.labels_ == 0, 0].mean(), X[model.labels_ == 0, 1].mean())
    c1_new = (X[model.labels_ == 1, 0].mean(), X[model.labels_ == 1, 1].mean())
    return c0_new, c1_new

calc_new_centroid(model1)



model2 = KMeans(n_clusters=2, init="random", n_init=1,
                max_iter=2, random_state=1).fit(X)


c0, c1 = model2.cluster_centers_
c0, c1

model2.score(X)

plot_cluster(model2, c0, c1)

calc_new_centroid(model2)



model3 = KMeans(n_clusters=2, init="random", n_init=1,
                max_iter=3, random_state=1).fit(X)

c0, c1 = model3.cluster_centers_
c0, c1


model3.score(X)


plot_cluster(model3, c0, c1)

calc_new_centroid(model3)





model4 = KMeans(n_clusters=2, init="random", n_init=1,
                max_iter=4, random_state=1).fit(X)

c0, c1 = model4.cluster_centers_
c0, c1

model4.score(X)


plot_cluster(model4, c0, c1)

calc_new_centroid(model4)




model5 = KMeans(n_clusters=2, init="random", n_init=1,
                max_iter=5, random_state=1).fit(X)

c0, c1 = model5.cluster_centers_
c0, c1

model5.score(X)

plot_cluster(model5, c0, c1)

kmeans_df(model5, c0, c1)

calc_new_centroid(model5)





from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


np.random.seed(5)
centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data
y = iris.target


def plot_iris_cluster(model):
    fig = plt.figure()
    fig.gca(projection='3d')
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    model.fit(X)
    labels = model.labels_
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(
        np.float), s=100, cmap=mpl.cm.jet)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    plt.show()


model3 = KMeans(n_clusters=3)
plot_iris_cluster(model3)





from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

digits = load_digits()
data = scale(digits.data)

def show_digits(images, labels):
    f = plt.figure(figsize=(10,2))
    f.gca(projection='3d')
    plt.subplots_adjust(top=1, bottom=0, hspace=0, wspace=0.05)
    i = 0
    while (i < 10 and i < images.shape[0]):
        ax = f.add_subplot(1, 10, i + 1)
        ax.imshow(images[i], cmap=plt.cm.bone)
        ax.grid(False)
        ax.table
        ax.set_title(labels[i])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.tight_layout()
        i += 1


show_digits(digits.images, range(10))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, images_train, images_test = \
    train_test_split(data, digits.target, digits.images,
                     test_size=0.25, random_state=42)

model = KMeans(init="k-means++", n_clusters=10, random_state=42)
model.fit(X_train)
show_digits(images_train, model.labels_)



y_pred = model.predict(X_test)

def show_cluster(images, y_pred, cluster_number):
    images = images[y_pred == cluster_number]
    y_pred = y_pred[y_pred == cluster_number]
    show_digits(images, y_pred)

for i in range(10):
    show_cluster(images_test, y_pred, i)


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)