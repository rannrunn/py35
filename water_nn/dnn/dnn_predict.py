# 텐서플로우를 사용해 2개 은닉층을 가진 DNN을 구성
# 각각 32, 16개의 신경을 가짐
#
# 오버피팅을 피하기 위해 드롭아웃을 적용
import numpy as np
import pandas as pd
import tensorflow as tf

# 로그가 많다면 아래 코드를 삭제
tf.logging.set_verbosity(tf.logging.INFO)


BATCH_SIZE = 2000
TRAIN_EPOCHS = 2000
# 입력값
columns = [
    "X1",
    "X2",
]


# 숫자형 입력값
X1 = tf.feature_column.numeric_column("X1")
X2 = tf.feature_column.numeric_column("X2")


# 특징을 담는 변수
feature_columns = {X1, X2}


# 정규화 진행
def normalize_column(col):
    return (col - np.mean(col)) / np.std(col)


# 인풋벡터 생성
def make_input_fn(filename):
    # CSV 파일을 읽어옴
    df = pd.read_csv("c:\\tmp\\csv\\sector02.csv")

    # 정규화 진행
    df["X1"] = normalize_column(df["X1"].values)
    df["X2"] = normalize_column(df["X2"].values)


    # 인풋 벡터
    def input_fn():
        # 유용한 특징을 배열로 변경
        useful_fueatures = [
            np.array(df["X1"].values, dtype=np.float32),
            np.array(df["X2"].values, dtype=np.float32),
            np.array(df["Y"].values, dtype=np.float32),
        ]

        # Ugly, 선택된 특징에 대해 슬라이스 인풋 프로듀서를 생성
        X1, X2, Y = tf.train.slice_input_producer(
            tensor_list=useful_fueatures,
            num_epochs=TRAIN_EPOCHS,
            shuffle=True,
            capacity=BATCH_SIZE * 5
        )

        # 슬라이스 인풋 프로듀서를 딕셔너리형식으로 변경
        dataset_dict = dict(
            X1=X1,
            X2=X2,
            Y=Y
        )

        # 데이터를 로드하는 배치 딕셔너리 큐를 생성
        # 멀티쓰레딩을 이용해 트레이닝을 진행
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

        # 라벨은 분리하여 반환
        batch_labels = batch_dict.pop('Y')
        return batch_dict, tf.reshape(batch_labels, [-1, 1])

    return input_fn


def make_model(features, labels, mode, params, config):
    # 인풋 레이어 생성
    input_layer = tf.feature_column.input_layer(
        features=features,
        feature_columns=feature_columns
    )

    # global step 을 가져옴
    global_step = tf.contrib.framework.get_or_create_global_step()

    # 첫번째 은닉층
    x = tf.layers.dense(
        inputs=input_layer,
        units=32,
        activation=tf.nn.relu,
        name="fisrt_fully_connected_layer"
    )

    # 오버피팅 방지를 위해 드롭아웃 추가
    x = tf.layers.dropout(
        inputs=x,
        name="first_dropout"
    )

    # 두번째(마지막) 은닉층
    x = tf.layers.dense(
        inputs=x,
        units=16,
        activation=tf.nn.relu,
        name="second_fully_connected_layer"
    )

    # 신경망
    predictions = tf.contrib.layers.fully_connected(
        inputs=x,
        num_outputs=1
    )

    # 손실함수는 아웃라이어에 덜 민감한 L1 거리로 정의
    loss = tf.losses.absolute_difference(
        labels=labels,
        predictions=predictions
    )

    # 텐서보드로 손실함수 EXPORT
    tf.summary.scalar("Loss", loss)

    # 향상된 기능과 안정성을 가진 아다그리드 모멘텀 옵티마이저를 사용
    optimizer = tf.train.AdamOptimizer(
        learning_rate=params.learning_rate,
    )

    # 텐서플로우 그래프에서 train_op를 출력함으로써 글로벌스템의 향상을 가져옴
    train_op = optimizer.minimize(loss, global_step=global_step)

    # 텐서를 정의
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op
    )


# 메인
def main(_):
    input_fn = make_input_fn(None)

    # 하이퍼파라미터 생성
    hparams = tf.contrib.training.HParams(
        learning_rate=.01,
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
        # 모델 디랙토리 생성
        model_dir=("/tmp/tf/water"),
        session_config=config
    )

    estimator = tf.estimator.Estimator(
        model_fn=make_model,
        params=hparams,
        config=trainingConfig
    )

    # 트레이닝 수행
    # estimator.train(
    #     input_fn=input_fn,
    #     steps=TRAIN_EPOCHS,
    # )


    y = estimator.predict(
        input_fn=input_fn
    )









# 메인 실행
if __name__ == '__main__':
    tf.app.run(main)