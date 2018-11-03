import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy


def get_predict_data(sw_id, df, idx, predict_period):

    list_df = []
    cnt = 0
    # 1074 -> 2018-05-28 00:00:00 (월요일 0시)
    for _ in range(5):
        df_item = pd.DataFrame(df['current_' + str(sw_id)])[idx + (cnt - 42): idx + cnt]
        df_item.reset_index()
        list_df.append(copy.deepcopy(df_item))
        cnt -= 42

    sum = 0
    for item in list_df:
        sum += item['current_' + str(sw_id)].values

    predict_data = (sum / len(list_df))[:predict_period]

    return predict_data


# time, real, train, predict
# flag_predict : 1. 4시간, 2. 하루, 3. 일주일
# JSON 형태로 리턴
def predict(sw_id, time, flag_predict):

    predict_period = 0

    df = pd.read_csv('c:/sim_data.csv')

    df.reset_index()
    df = df.rename(columns = {'Unnamed: 0':'time_id'})
    df['time_id'] = pd.to_datetime(df['time_id'])
    # dayofweek -> 0:월요일, 6:일요일
    df['weekday'] = df['time_id'].dt.dayofweek

    mask = df['time_id'] == time
    idx = df[mask].index.values[0]

    if idx < 210:
        print('idx error')
        return {
            'time':[]
            , 'real':[]
            , 'train':[]
            , 'pred':[]
            , 'error':'idx error'
        }

    if flag_predict == '1':
        predict_period = 6
    elif flag_predict == '2':
        predict_period = 18
    elif flag_predict == '3':
        predict_period = 42

    predict_data = get_predict_data(sw_id, df, idx, predict_period)

    list_time = df.loc[idx - 126: idx + predict_period - 1]['time_id'].values

    dict_result = {
        'time':list_time
        , 'real':df['current_' + str(sw_id)][idx - 126: idx + predict_period].values
        , 'train':df['current_' + str(sw_id)][idx - 126: idx].values
        , 'pred':predict_data
        , 'error':''
    }

    return dict_result


dict_result = predict('2409', '2018-01-04 00:00:00', '3')

plt.plot(dict_result['time'], dict_result['real'], 'b')
plt.plot(dict_result['time'], np.append(dict_result['train'], dict_result['pred']), 'r')
plt.show()

