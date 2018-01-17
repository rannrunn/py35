# 0. 사용할 패키지 불러오기
import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
import pandas as pd
import time


df_x_train = pd.read_csv('./x_train_modify.csv', header=None)
df_y_train = pd.read_csv('./y_train_modify.csv', header=None)

# 1. 데이터셋 생성하기
x_test = np.round(((df_x_train.values - 35) / 70.0), 2)[4000:]
print(type(x_test[0]))
y_test = df_y_train.values[4000:]
# print(x_train)
# print(x_test)
xhat_idx = np.random.choice(x_test.shape[0], 100)
xhat = x_test[xhat_idx]

print(type(x_test))
print(np.shape(xhat[0]))

from keras.models import load_model
model = load_model('iot_pole_unbalance_load_model.h5')
start = time.time()
yhat = model.predict_classes(xhat)
print(time.time() - start)
cnt = 0
num = 0
for i in range(100):
    num += 1
    if str(y_test[xhat_idx[i]]) == str(yhat[i]):
        cnt += 1

    print(df_x_train.values[4000:][xhat_idx[i]][:24], df_x_train.values[4000:][xhat_idx[i]][24:48], df_x_train.values[4000:][xhat_idx[i]][48:72])

    plt.figure(figsize=(10, 15))
    plt.title('True : ' + str(y_test[xhat_idx[i]]) + ', Predict : ' + str(yhat[i]))
    plt.plot(df_x_train.values[4000:][xhat_idx[i]][:24])
    plt.plot(df_x_train.values[4000:][xhat_idx[i]][24:48])
    plt.plot(df_x_train.values[4000:][xhat_idx[i]][48:72])
    plt.ylim(-10, 60)

    plt.show()


