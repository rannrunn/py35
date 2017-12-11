import tensorflow as tf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# X and Y data
x1_data = [1.0, 2.0, 3.0]
x2_data  = [2.0, 3.0, 4.0]
y_data  = [1.0, 2.0, 3.0]

W1 = tf.Variable(tf.random_normal([1]), name='weight_01', dtype=tf.float32)
W2 = tf.Variable(tf.random_normal([1]), name='weight_02', dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), name='bias', dtype=tf.float32)


# Our hypothesis XW+b
hypothesis = x1_data * W1 + x2_data * W2 + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(5000):
   sess.run(train)
   if step % 20 == 0:
       print(step, sess.run(cost), sess.run(W1), sess.run(W2), sess.run(b))

# list 형식으로 변수 저장 : 하지만 쓰지 않는다.(list와 list는 곱할 수 없기 때문에)
weights_1 = []
weights_1.append(W1)
weights_2 = []
weights_2.append(W2)
bias = []
bias.append(b)

# list 와 numpy.ndarray는 서로 곱할 수 있다.
# list 와 list 는 곱할 수 없다.
# W1.eval(sess) 는 numpy.ndarray 형식의 값을 리턴한다.
z = x1_data * W1.eval(sess) + x2_data * W2.eval(sess) + b.eval(sess)

print(type(x1_data)) # list
print(type(W1.eval(sess))) # numpy.ndarray
print(type(x2_data)) # list
print(type(W2.eval(sess))) # numpy.ndarray
print(type(b.eval(sess))) # numpy.ndarray

print(z)
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1_data, x2_data, y_data, 'b')
ax.plot(x1_data, x2_data, z, 'r')
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_zlim(0, 5)

plt.show()


