import matplotlib.pyplot as plt
import ruptures as rpt
import os
import pandas as pd
from matplotlib import gridspec
import time
from multiprocessing import Pool


def pelt_plot(filepath):

    dir_output = 'C:\\_data\\pqms_change_detection_pelt_plot\\'

    if not os.path.isdir(dir_output):
        os.makedirs(dir_output)

    filename = os.path.basename(filepath)
    dirname = os.path.dirname(filepath)

    if not filename.find('3상') > -1: #
        return

    df_temp = pd.read_excel(filepath)
    df_temp = df_temp.rename(columns={'Unnamed: 1': 'load'})

    df_temp.set_index(pd.to_datetime(df_temp[df_temp.columns[0]]), inplace=True)
    df_4H_mean = df_temp.resample('4H').mean()
    df_D_mean = df_temp.resample('D').mean()

    arr = df_D_mean['load'].values

    start = time.time()

    algo = rpt.Pelt(model="rbf").fit(arr)
    result = algo.predict(pen=10)


    plt.figure(figsize=(7,5))

    print('시간:', time.time() - start)

    rpt.display(arr, result, result)
    plt.tight_layout()
    plt.savefig(dir_output + filepath.replace('C:\\_data\\부하데이터\\', '').replace('\\', '_').replace('.xls', '.png'), bbox_inches='tight')
    # plt.show()
    plt.close()


if __name__ == '__main__':

    dir_source = 'C:\\_data\\부하데이터'

    names = []
    for path, dirs, filenames in os.walk(dir_source):
        for filename in filenames:
            names.append(os.path.join(path, filename))

    print(names)

    with Pool(processes=14) as pool:
        pool.map(pelt_plot, names)

