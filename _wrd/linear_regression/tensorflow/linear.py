import numpy as np
import math

num_points = 1000
vectors_set = []
for i in range(num_points):
         x1= np.random.normal(-2000, 0.55)

         print(x1)
         y1= x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
         vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

print(x_data)

import matplotlib.pyplot as plt

#Graphic display
plt.plot(x_data, y_data, 'ro', label='data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

import tensorflow as tf

W = tf.Variable(tf.random_uniform([1], 0.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = tf.sqrt(W * x_data) + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess = tf.Session()
    sess.run(init)

    for step in range(15000):
         sess.run(train)
         print(step, sess.run(W), sess.run(b))
         print(step, sess.run(loss))

         if step % 1000 == 0:
             print('xx:', step)

         if (step == 14999):
             #Graphic display
             plt.plot(x_data, y_data, 'ro', label='data')
             plt.plot(x_data, np.sqrt(sess.run(W) * x_data) + sess.run(b), label='line')
             plt.xlabel('x')
             plt.ylabel('y')
             #plt.xlim(-2,2)
             #plt.ylim(0.1,0.6)
             plt.legend()
             plt.show()
