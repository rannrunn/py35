import pandas as pd
import numpy as np
import seaborn as sns
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt



num_cores = 4
iris = pd.DataFrame(sns.load_dataset('iris'))

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def multiply_columns(data):
    time.sleep(np.random.randint(10))
    data['length_of_word'] = data['species'].apply(lambda x: len(x))
    print(data)
    return data



if __name__ == '__main__':

    start_time = time.time()

    normal_iris = multiply_columns(iris)

    print('시간1:', time.time() - start_time)



    start_time = time.time()

    parallel_iris = parallelize_dataframe(iris, multiply_columns)

    print('시간2:', time.time() - start_time)

    print(parallel_iris.to_string())






