import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_predict_data(df, idx):
    # 인덱스를 결정해서 데이터를 불러오는 시점을 결정
    # 시간 계산
    idx = 1024
    df_before = df['current_5576'][idx - 84: idx]

    df_42 = pd.DataFrame(df['current_5576'])[idx - 42: idx]
    df_42.reset_index()

    df_84 = pd.DataFrame(df['current_5576'])[idx - 84: idx - 42]
    df_84.reset_index()

    df_126 = pd.DataFrame(df['current_5576'])[idx - 126: idx - 84]
    df_126.reset_index()

    predict = (df_42['current_5576'].values + df_84['current_5576'].values + df_126['current_5576'].values) / 3
    return predict

# time, real, train, predict
# flag_predict : 1. 4시간, 2. 하루, 3. 일주일
# JSON 형태로 리턴
def predict(sw_id, time, flag_predict):
    df = pd.read_csv('c:/sim_data.csv')
    df.reset_index()

    idx = 1024
    predict_data = get_predict_data(df, idx)


    plt.plot(predict_data, 'r')
    plt.show()

    plt.plot(df['current_5576'][idx - 126: idx].values, 'r')
    plt.show()


    plt.plot(df['current_5576'][idx - 126: idx + 42].values, 'b')
    plt.plot(np.append(df['current_5576'][idx - 126: idx].values, predict_data), 'r')
    plt.show()


predict('5576', '2017-12-30', '3')
