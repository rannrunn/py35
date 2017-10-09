# Logistic regression example in TF using Kaggle's Titanic Dataset.
# Download train.csv from https://www.kaggle.com/c/titanic/data

import tensorflow as tf
import os

# same params and variables initialization as log reg.
W = tf.Variable(tf.zeros([24, 1]), name="weights")
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
    tag, time, f_paldang, f_gwangam, f_tie36, f_lottein, f_lotteout, f_yangje, f_gwachunin, f_paldangstl, f_gwachunoutm, f_hong1, f_hong2, f_hong3, f_sungsan, f_kimpo, p_paldangstl, p_gwangam, p_seohanam, p_oryun, p_songpa, p_sinchun, p_hong, p_lottein, p_gwachunyangje, p_gwachunout = \
        read_csv(100, "c:/_data/wrd_total_del_badrow.csv", [[0.0], [""], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

    # convert categorical data
    #is_first_class = tf.to_float(tf.equal(pclass, [1]))

    # Finally we pack all the features in a single matrix;
    # We then transpose to have a matrix with one example per row and one feature per column.
    features = tf.transpose(tf.stack([f_paldang, f_gwangam, f_tie36, f_lottein, f_lotteout, f_yangje, f_gwachunin, f_paldangstl, f_gwachunoutm, f_hong1, f_hong2, f_hong3, f_sungsan, f_kimpo, p_paldangstl, p_gwangam, p_seohanam, p_oryun, p_songpa, p_sinchun, p_hong, p_lottein, p_gwachunyangje, p_gwachunout]))
    survived = tf.reshape(tag, [100, 1])

    return features, survived


def train(total_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):

    predicted = tf.cast(inference(X) > 0.5, tf.float32)
    print('xxxxxxxxx')
    print (sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))

# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:

    tf.initialize_all_variables().run()

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # actual training loop
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        # for debugging and learning purposes, see how the loss gets decremented thru training steps
        if step % 10 == 0:
            print ("loss: ", sess.run([total_loss]))

    evaluate(sess, X, Y)

    import time
    time.sleep(5)

    coord.request_stop()
    coord.join(threads)
    sess.close()
