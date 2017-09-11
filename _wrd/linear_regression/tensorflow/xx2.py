import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

data = np.loadtxt("dataset.txt", unpack=True, dtype="float32")

x_data = np.transpose(data[0:-1])
y_data = np.transpose(data[-1])

y_data = np.reshape(y_data, [len(y_data), 1])

X = tf.placeholder("float", [None, 2])
Y = tf.placeholder("float", [None, 1])

w = tf.Variable(tf.random_uniform([2, 1], -2.0, 2.0))
b = tf.Variable(tf.random_uniform([1, 1], -2.0, 2.0))

y = tf.matmul(X, w) + b

loss = tf.reduce_mean(tf.square(y - Y))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(1001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 100 == 0:
        print (step, sess.run(w), sess.run(b))
        print ("loss:", sess.run(loss, feed_dict={X: x_data, Y: y_data}))

# Graph
fig = plt.figure()

x1, x2 = np.meshgrid(np.arange(-10, 10), np.arange(-10, 10))

x = np.transpose([x1.flatten(), x2.flatten()])

result = sess.run(y, feed_dict={X: x})

a1 = fig.add_subplot(111, projection="3d")
a1.scatter(data[0], data[1], data[2], c='r')
# 원함수 출력 원할 경우
#a1.plot_surface(x1, x2, 10 * x1 - 4 * x2 + 7, linewidth=0, rstride=1, cstride=1, cmap=cm.Blues)
a1.plot_surface(x1, x2, np.reshape(result, x1.shape), linewidth=0, rstride=1, cstride=1, cmap=cm.Greens)

a1.set_xlabel("X1(X)")
a1.set_ylabel("X2(Y)")
a1.set_zlabel("Y(Z)")

plt.show()

