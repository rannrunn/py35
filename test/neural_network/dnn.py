# Tensorflow custom implementation of a DNN consisting of 3 hidden layers of
# 512, 128, and 16 neurons respectively.
#
# The network is fed by the new tensorflow `input_fn` approach. This code demonstrates
# how to create a multithreaded input producer for batch training.
#
# Also, this DNN makes use of dropout to avoid overfitting.
#
# The training error converge to around 12.000$ after 80.000 training epochs.
import numpy as np
import pandas as pd
import tensorflow as tf

# Remove if you consider it too much
tf.logging.set_verbosity(tf.logging.INFO)

BATCH_SIZE = 2000
TRAIN_EPOCHS = 80000
columns = [
    "id",
    "date",
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15"
]


# General numeric features
bedrooms_feature = tf.feature_column.numeric_column("bedrooms")
sqft_living_feature = tf.feature_column.numeric_column("sqft_living")
sqft_lot_feature = tf.feature_column.numeric_column("sqft_lot")
bathrooms_x_bedrooms = tf.feature_column.numeric_column("bathrooms_x_bedrooms")  # This is a custom col


# finally create the set of all the columns
feature_columns = {bedrooms_feature, sqft_living_feature, sqft_lot_feature,
                   bathrooms_x_bedrooms}


def normalize_column(col):
    return (col - np.mean(col)) / np.std(col)


def make_input_fn(filename):
    # Read the csv
    df = pd.read_csv("../input/kc_house_data.csv")

    # Create custom columns
    df["bathrooms_x_bedrooms"] = list(map(lambda bed, bath: bed / bath if bath != 0 else 1.5,
                                          df["bedrooms"].values,
                                          df["bathrooms"].values))

    # Normalizing numerical features
    df["bedrooms"] = normalize_column(df["bedrooms"].values)
    df["bathrooms_x_bedrooms"] = normalize_column(df["bathrooms_x_bedrooms"].values)
    df["sqft_living"] = normalize_column(df["sqft_living"].values)
    df["sqft_lot"] = normalize_column(df["sqft_lot"].values)

    # Remove unused columns
    del df["id"]
    del df["date"]
    del df["zipcode"]
    del df["yr_built"]
    del df["lat"],
    del df["long"],
    del df["view"],
    del df["condition"],
    del df["waterfront"],
    del df["grade"],

    # Create tensorflow input fn
    def input_fn():
        # Wrap the useful features in an array
        useful_fueatures = [
            np.array(df["bedrooms"].values, dtype=np.float32),
            np.array(df["sqft_living"].values, dtype=np.float32),
            np.array(df["sqft_lot"].values, dtype=np.float32),
            np.array(df["bathrooms_x_bedrooms"].values, dtype=np.float32),
            np.array(df["price"].values, dtype=np.float32)
        ]

        # Ugly, but creates all the slice input producers for all the features selected
        bedrooms, sqft_living, \
        sqft_lot, bathrooms_x_bedrooms, \
        labels = tf.train.slice_input_producer(
            tensor_list=useful_fueatures,
            num_epochs=TRAIN_EPOCHS,
            shuffle=True,
            capacity=BATCH_SIZE * 5
        )

        # Created a dict out of sliced input producers
        dataset_dict = dict(
            bedrooms=bedrooms,
            sqft_living=sqft_living,
            sqft_lot=sqft_lot,
            bathrooms_x_bedrooms=bathrooms_x_bedrooms,
            labels=labels
        )

        # Creates a batched dictionary that holds a queue that loads the data
        # while the training is happening. Multithreading.
        batch_dict = tf.train.batch(
            dataset_dict,
            BATCH_SIZE,
            num_threads=10,
            capacity=BATCH_SIZE * 5,
            enqueue_many=False,
            dynamic_pad=False,
            allow_smaller_final_batch=True,
            shared_name=None,
            name=None
        )

        # The labels need to be returned separately
        batch_labels = batch_dict.pop('labels')
        return batch_dict, tf.reshape(batch_labels, [-1, 1])

    return input_fn


def make_model(features, labels, mode, params, config):
    # Creates the input layer starting from the feature definitions of above
    input_layer = tf.feature_column.input_layer(
        features=features,
        feature_columns=feature_columns
    )

    # Get the global step
    global_step = tf.contrib.framework.get_or_create_global_step()

    # First dense layer or the neural net
    x = tf.layers.dense(
        inputs=input_layer,
        units=512,
        activation=tf.nn.relu,
        name="fisrt_fully_connected_layer"
    )

    # Adding dropout to lessen chances of overfitting
    x = tf.layers.dropout(
        inputs=x,
        name="first_dropout"
    )

    # Second dense layer
    x = tf.layers.dense(
        inputs=x,
        units=128,
        activation=tf.nn.relu,
        name="second_fully_connected_layer"
    )

    # Third and final deep layer of the neural net
    x = tf.layers.dense(
        inputs=x,
        units=16,
        activation=tf.nn.relu,
        name="third_fully_connected_layer"
    )

    # Linear output neuron that combine the output of the neural net
    predictions = tf.contrib.layers.fully_connected(
        inputs=x,
        num_outputs=1
    )

    # Loss is defined as the L1 distance since it is less sensitive to outliers
    loss = tf.losses.absolute_difference(
        labels=labels,
        predictions=predictions
    )

    # Export the loss to tensorboard
    tf.summary.scalar("Loss", loss)

    # Using ADAgrad Momentum Optimizer since it provides quite some advance features and
    # turns out to be very stable
    optimizer = tf.train.AdamOptimizer(
        learning_rate=params.learning_rate,
    )

    # Out train op in the tensorflow graph. Computing this also increases our global_step
    train_op = optimizer.minimize(loss, global_step=global_step)

    # Finally, wrap the tensor defined above in the format Tensorflow expects
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op
    )


# Main file
def main(_):
    input_fn = make_input_fn(None)

    # Creates hyperparams
    hparams = tf.contrib.training.HParams(
        learning_rate=.1,
    )

    config = tf.ConfigProto(
        # allow_soft_placement=True,
        # log_device_placement=True
    )
    # Turns on JIT Compilation through XLA for boosting performance. If crashes disable this
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    trainingConfig = tf.contrib.learn.RunConfig(
        # log_device_placement=True,
        save_summary_steps=500,
        save_checkpoints_steps=500,
        # Creates model dir (need to change this)
        model_dir=("/tmp/tf-logs/bucketized-04"),
        session_config=config
    )

    estimator = tf.estimator.Estimator(
        model_fn=make_model,
        params=hparams,
        config=trainingConfig
    )

    # Finally, perform the training (VERY VERY LONG!)
    estimator.train(
        input_fn=input_fn,
        steps=TRAIN_EPOCHS,
    )


# Run the main
if __name__ == '__main__':
    tf.app.run(main)