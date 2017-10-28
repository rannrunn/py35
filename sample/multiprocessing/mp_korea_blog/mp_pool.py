import os
import multiprocessing as mp

def doubler(number):
    result = number * 2
    proc = os.getpid()
    print('{0} doubled to {1} by process id: {2}'.format( number, result, proc))

if __name__ == '__main__':
    numbers = [5, 10, 15, 20, 25]
    procs = []

    p = mp.Pool(4)
    p.map(doubler, numbers)
