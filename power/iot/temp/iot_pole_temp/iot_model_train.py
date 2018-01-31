# 0. 사용할 패키지 불러오기
import numpy as np
np.set_printoptions(threshold=np.nan)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import random
import pandas as pd


df_x_train = pd.read_csv('./x_train_modify.csv', header=None)
df_y_train = pd.read_csv('./y_train_modify.csv', header=None)


# 1. 데이터셋 생성하기
x_train = df_x_train.values
x_test = np.round(((x_train - 35) / 70.0), 2)[4000:]
x_train = np.round(((x_train - 35) / 70.0), 2)[:4000]
y_train = df_y_train.values[:4000]
y_test = df_y_train.values[4000:]

print(np.shape(x_train))
print(np.shape(y_train))

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(144, input_dim=72, activation='relu'))
model.add(Dense(72, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 모델 학습과정 설정하기
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=200, batch_size=72)

# 5. 학습과정 살펴보기
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.set_ylim([0.0, 1.0])
acc_ax.set_ylim([0.0, 1.0])

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
acc_ax.plot(hist.history['acc'], 'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('loss_and_metrics : ' + str(loss_and_metrics))


from keras.models import load_model
model.save('iot_pole_unbalance_load_model.h5')