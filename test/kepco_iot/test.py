import os
import multiprocessing as mp

def f(x):
    print(x * x)
    return x * x

print(__name__)
if __name__ == '__main__':
    coreNum = mp.cpu_count() # check the number of CPU cores
    coreNum_toUse = coreNum - 1
    print("coreNum_toUse: ", coreNum_toUse)
    mp_ = mp.Pool(coreNum_toUse)
    d = mp_.map(f, [1, 2, 3])

