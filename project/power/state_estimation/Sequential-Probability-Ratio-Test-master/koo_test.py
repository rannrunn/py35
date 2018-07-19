import sprt as sprt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_pqms_decomposition():
    dir_source = 'C:\\_data\\부하데이터'
    dir_output = 'C:\\_data\\pqms_load_sprt_plot_one\\'

    if not os.path.isdir(dir_output):
        os.makedirs(dir_output)

    filepath = 'C:\_data\부하데이터\수원경기본부\금촌변전소\ID5\금촌_3상.xls'
    root = os.path.dirname(filepath)
    file = os.path.basename(filepath)


    df_temp = pd.read_excel('C:\_data\부하데이터\수원경기본부\금촌변전소\ID5\금촌_3상.xls')
    df_temp = df_temp.rename(columns={'Unnamed: 1': 'load'})


    df_temp.set_index(pd.to_datetime(df_temp[df_temp.columns[0]]), inplace=True)
    df_temp = df_temp.resample('D').mean().interpolate('linear')
    test = sprt.SPRTNormal(alpha = 0.05, beta = 0.2, h0 = 0, h1 = 1,
                              values = df_temp['load'],
                              variance = 1)

    test.plot()
    plt.xticks(rotation=30)
    plt.legend()
    plt.show()
    plt.close()





if __name__ == '__main__':
    plot_pqms_decomposition()



import sprt as sprt
import numpy as np

data = np.random.normal(0, 1, 10)

test = sprt.SPRTNormal(alpha = 0.05, beta = 0.2, h0 = 0, h1 = 1,
                       values = data,
                       variance = 1)

test.plot()
plt.show()