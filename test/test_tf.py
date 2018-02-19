import tensorflow as tf



# Build a dataflow graph.
c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
e = tf.matmul(c, d)

print(type(c))

# x = tf.assign(c, [[1.0, 1.0], [0.0, 1.0]])



with tf.Session() as sess:



    # Create a Variable, that will be initialized to the scalar value 0.
    state = tf.Variable(2, name="counter")
    init_op = tf.initialize_all_variables()

    sess.run(init_op)
    print(type(state))
    print(state)
    print(sess.run(state))
    # Variable을 생성하는 것 또한 노드 (op) 다. # Create an Op to add one to `state`.
    one = tf.constant(1)
    print(one)
    print('type one:',type(one))
    new_value = tf.add(state, one)
    print(state)
    print(sess.run(state))
    update = tf.assign(state, new_value)
    print(type(update))
    print(update)
    print(sess.run(update))

    a = tf.constant([[11,0,13,14],
                     [21,22,23,0]])
    condition = tf.zeros(a)
    print(condition.eval())
    case_true = tf.reshape(tf.multiply(tf.ones([8], tf.int32), -9999), [2, 4])
    case_false = a
    a_m = tf.where(condition, case_true, case_false)


