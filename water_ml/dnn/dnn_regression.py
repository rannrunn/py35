# 텐서플로우를 사용해 2개 은닉층을 가진 DNN을 구성
# 각각 32, 16개의 신경을 가짐
#
# 오버피팅을 피하기 위해 드롭아웃을 적용
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# 로그가 많다면 아래 코드를 삭제
tf.logging.set_verbosity(tf.logging.INFO)


dict_section = {}
dict_section['section01'] = '4'
dict_section['section02'] = '2'
dict_section['section03'] = '3'
dict_section['section04'] = '2'
dict_section['section05'] = '1'
dict_section['section06_01'] = '3'
dict_section['section06_02'] = '1'
dict_section['section06_03'] = '1'
dict_section['section07_01'] = '1'
dict_section['section07_02'] = '1'
dict_section['section07_03'] = '3' # 양재23T의 압력이 없어서 추후 시도할 수 있음


# 옵션 : train, predict
option = 'predict'
# section
section = 'section07_02'
# 데이터프레임 구간 사용유무
bool_section = False
# 데이터프레임 시작지점
df_start = 0
# 데이터프레임 종료지점
df_end = 1000
# 한번의 학습 시 사용하는 데이터의 수로 전체 데이터 수보다 작거나 같음
BATCH_SIZE = 2000
# 학습 횟수
TRAIN_EPOCHS = 8000
if option == 'predict':
    TRAIN_EPOCHS = 1
# 섞는 정도

# 추론일 경우 섞지 않고 멀티 쓰레드도 사용하지 않는다.
# 둘을 사용할 경우 섞여서 결과가 다르게 나옴
shuffle = True
num_threads = 10
if option == 'predict':
    shuffle = False
    num_threads = 1
print('shuffle:', shuffle)
# 입력값
columns = [
    "x1",
    "x2",
    "x3",
    "x4",
]

# 숫자형 입력값
x1 = tf.feature_column.numeric_column("x1")
x2 = tf.feature_column.numeric_column("x2")
x3 = tf.feature_column.numeric_column("x3")
x4 = tf.feature_column.numeric_column("x4")



# 특징을 담는 변수
feature_columns = set({})
for i in range(1, int(dict_section[section]) + 1):
    feature_columns.add(eval('x' + str(i)))


# 정규화 진행
def normalize_column(col):
    return (col - np.mean(col)) / np.std(col)


# 인풋벡터 생성
def make_input_fn(filename):
    # 데이터프레임을 전역변수로 선언
    global df
    # CSV 파일을 읽어옴
    df = pd.read_csv('c:\\tmp\\csv\\' + section + '.csv')

    # 정규화 진행
    for i in range(1, int(dict_section[section]) + 1):
        df["x" + str(i)] = normalize_column(df["x" + str(i)].values)

    if bool_section == True:
        df = df.iloc[df_start:df_end,:]

    # 인풋 벡터
    def input_fn():
        # 유용한 특징을 배열로 변경
        useful_fueatures = []
        for i in range(1, int(dict_section[section]) + 1):
            useful_fueatures.append(np.array(df["x" + str(i)].values, dtype=np.float32))
        useful_fueatures.append(np.array(df["y"].values, dtype=np.float32))


        x1 = None
        x2 = None
        x3 = None
        x4 = None
        y = None
        # Ugly, 선택된 특징에 대해 슬라이스 인풋 프로듀서를 생성
        if dict_section[section] == '1':
            x1, y = tf.train.slice_input_producer(
                tensor_list=useful_fueatures,
                num_epochs=TRAIN_EPOCHS,
                shuffle=shuffle,
                capacity=BATCH_SIZE * 5
            )
        elif dict_section[section] == '2':
            x1, x2, y = tf.train.slice_input_producer(
                tensor_list=useful_fueatures,
                num_epochs=TRAIN_EPOCHS,
                shuffle=shuffle,
                capacity=BATCH_SIZE * 5
            )
        elif dict_section[section] == '3':
            x1, x2, x3, y = tf.train.slice_input_producer(
                tensor_list=useful_fueatures,
                num_epochs=TRAIN_EPOCHS,
                shuffle=shuffle,
                capacity=BATCH_SIZE * 5
            )
        elif dict_section[section] == '4':
            x1, x2, x3, x4, y = tf.train.slice_input_producer(
                tensor_list=useful_fueatures,
                num_epochs=TRAIN_EPOCHS,
                shuffle=shuffle,
                capacity=BATCH_SIZE * 5
            )


        # 슬라이스 인풋 프로듀서를 딕셔너리형식으로 변경
        dataset_dict = {}
        for i in range(1, int(dict_section[section]) + 1):
            dataset_dict['x' + str(i)] = eval('x' + str(i))
        dataset_dict['y'] = y


        # 데이터를 로드하는 배치 딕셔너리 큐를 생성
        # 멀티쓰레딩을 이용해 트레이닝을 진행
        batch_dict = tf.train.batch(
            dataset_dict,
            BATCH_SIZE,
            num_threads=num_threads,
            capacity=BATCH_SIZE * 5,
            enqueue_many=False,
            dynamic_pad=False,
            allow_smaller_final_batch=True,
            shared_name=None,
            name=None
        )

        # 라벨은 분리하여 반환
        batch_labels = batch_dict.pop('y')
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

    # 트레이닝 상황일 경우 오버피팅 방지를 위해 드롭아웃 추가
    if option == 'train':
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


    loss = None
    optimizer = None
    train_op = None
    # estimator.predict를 사용할 경우 labels 이 입력되지 않아 에러가 나기 때문에 따로 처리
    if labels is not None:
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
        predictions={"y":predictions},
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
        # 모델 디랙토리 생성 : 체크포인트 등이 저장됨
        model_dir=("c:/tmp/tf/water/" + section),
        session_config=config
    )

    estimator = tf.estimator.Estimator(
        model_fn=make_model,
        params=hparams,
        config=trainingConfig
    )

    if option == 'train':
        # 트레이닝 수행
        estimator.train(
            input_fn=input_fn,
            steps=TRAIN_EPOCHS,
        )
    elif option == 'predict':
        pred = estimator.predict(input_fn=input_fn)
        list_pred = []
        for i, p in enumerate(pred):
            #print("Prediction %s: %s" % (i + 1, p["y"]))
            list_pred.append(p["y"][0])
        df['pred'] = list_pred
        df['diff'] = df['y'] - df['pred']
        print(df['pred'])
        print(df['y'])
        # 평균절대값 오차 계산
        print(df['diff'].abs().mean())

        # 라벨값과 추론값의 상관관계 계산
        print(df['y'].corr(df['pred']))

        plt.figure(figsize=(12, 6))
        x = range(0, len(df))
        y = df['y'].values
        y_pred = df['pred'].values
        plt.plot(x, y, 'ro', label='Actual')
        plt.plot(x, y_pred, 'b', label='Prediction')
        plt.legend()
        plt.show()



# 메인 실행
if __name__ == '__main__':
    tf.app.run(main)

