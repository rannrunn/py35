import tensorflow as tf


x1_data = [73., 93., 89., 96., 73.]

x2_data = [80., 88., 91., 98., 66.]

x3_data = [75., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142.]



x1 = tf.placeholder(tf.float32)

x2 = tf.placeholder(tf.float32)

x3 = tf.placeholder(tf.float32)



Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')

w2 = tf.Variable(tf.random_normal([1]), name='weight2')

w3 = tf.Variable(tf.random_normal([1]), name='weight3')

b = tf.Variable(tf.random_normal([1]), name='bias')


x_data = [[0.0073, 0.0080, 0.0075],

          [0.0093, 0.0088, 0.0093],

          [0.0089, 0.0091, 0.0090],

          [0.0096, 0.0098, 0.00100],

          [0.0073, 0.0066, 0.0070]]

y_data = [[152.],

          [185.],

          [180.],

          [196.],

          [142.]]

X = tf.placeholder(tf.float32, shape=[None, 3])

Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')

b = tf.Variable(tf.random_normal([1]), name='bias')


# 가설 정의

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)

train = optimizer.minimize(cost)

sess = tf.Session()

# 변수 초기화

sess.run(tf.global_variables_initializer())



for step in range(100001):

    # 위에서 정의한 행렬 데이터 셋을 placeholder 변수에 입력한다.

    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})

    if step % 2000 == 0:

        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
