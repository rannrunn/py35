# 텐서플로우를 사용할 것을 알려줍니다.
import tensorflow as tf
# x_data는 공부시간, y_data는 시험성적 입니다.
x_data = [4, 9, 16]
y_data = [2, 3, 4]
# W는 설명변수, b는 보정값 입니다.
W = tf.Variable(tf.random_uniform([1], 0, 1.0))
b = tf.Variable(tf.random_uniform([1], 0, 1.0))
# placeholder를 사용하면 변수의 형태만 지정해주고 나중에 값을 넣어줘도 됩니다.
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
# 방정식 모델입니다.
hypothesis = tf.sqrt(W * X) + tf.sqrt(W * X) +b
# cost 함수입니다. 뒤어서 설명하겠습니다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Gradient Descent 또한 뒤에서 설명하겠습니다.
a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
# 변수들을 초기화합니다.
init = tf.global_variables_initializer()
# 텐서플로우를 시작하게 하는 구문이라고 보시면 됩니다. 세션을 지정해줍니다.
sess = tf.Session()
sess.run(init)
print(sess.run(W))
print(sess.run(b))
# X에 x_data를, Y에 y_data를 넣어서 2001번 소스를 돌려가며 W와 b값을 찾아갑니다.
for step in range(15000):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 200 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))
# 5시간 공부했을때와 2.5시간 공부했을 때 몇점이 나올지 출력해봅니다.
print(sess.run(hypothesis, feed_dict={X: 5}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))