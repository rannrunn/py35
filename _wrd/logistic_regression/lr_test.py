# Logistic regression example in TF using Kaggle's Titanic Dataset.
# Download train.csv from https://www.kaggle.com/c/titanic/data

import tensorflow as tf
import os

# same params and variables initialization as log reg.
W = tf.Variable(tf.zeros([2, 1]), name="weights")
b = tf.Variable(0., name="bias")


# former inference is now used for combining inputs
def combine_inputs(X):
    return tf.matmul(X, W) + b


# new inferred value is the sigmoid applied to the former
def inference(X):
    return tf.sigmoid(combine_inputs(X))


def loss(X, Y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))


def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.join(os.getcwd(), file_name)])

    print('filename_queue = ', filename_queue)

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)


def inputs():
    a, b, c = \
        read_csv(100, "test.csv", [[0.0], [0.0], [0.0]])

    # convert categorical data

    # Finally we pack all the features in a single matrix;
    # We then transpose to have a matrix with one example per row and one feature per column.
    features = tf.transpose(tf.stack([a, b]))
    water_leak = tf.reshape(c, [100, 1])

    return features, water_leak


def train(total_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):

    predicted = tf.cast(inference(X) > 0.5, tf.float32)
    print (sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))

# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:

    # Add ops to save and restore all the variables.

    option = 'predict'

    X, Y = inputs()
    saver = tf.train.Saver()

    if (option == 'start'):
        tf.initialize_all_variables().run()
    elif (option == 'add' or option == 'predict'):
        # Restore variables from disk.
        saver.restore(sess, "./ckpt/logistic_regression_test.ckpt")
        print("Model restored.")

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # actual training loop
    if(option != 'predict'):
        training_steps = 1000
        for step in range(training_steps):
            sess.run([train_op])
            # for debugging and learning purposes, see how the loss gets decremented thru training steps
            if step % 10 == 0:
                print("loss: ", sess.run([total_loss]))

    evaluate(sess, X, Y)

    # Save the variables to disk.
    save_path = saver.save(sess, "./ckpt/logistic_regression_test.ckpt")
    print("Model saved in file: %s" % save_path)

    import time

    time.sleep(5)

    coord.request_stop()
    coord.join(threads)
    sess.close()