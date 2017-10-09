import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# z = 10x - 4y + 7

#draw 3D graph
fig = plt.figure()

x, y = np.meshgrid(np.arange(-10, 10), np.arange(-10, 10))

a1 = fig.add_subplot(111, projection="3d")
a1.plot_surface(x, y, 10 * x - 4 * y + 7, rstride=1, cstride=1, cmap=cm.Blues)

a1.set_xlabel("X")
a1.set_ylabel("Y")
a1.set_zlabel("Z")

plt.show()

# data maker
fig = plt.figure()

coordinates = []

for i in range(100):
    x = np.random.normal(0.0, 1.)
    y = np.random.normal(0.0, 1.0)
    z = 10 * x - 4 * y + 7 +  np.random.normal(0.0, 0.1)
    coordinates.append([x, y, z])

a2 = fig.add_subplot(111, projection="3d")
a2.scatter([v[0] for v in coordinates], [v[1] for v in coordinates], [v[2] for v in coordinates])

a2.set_xlabel("X")
a2.set_ylabel("Y")
a2.set_zlabel("Z")

plt.show()

np.savetxt("dataset.txt", coordinates, fmt="%f")

