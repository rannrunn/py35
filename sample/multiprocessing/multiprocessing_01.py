import os
from multiprocessing import Process

def doubler(number):
    result = number * 2
    proc = os.getpid()
    print('{0} doubled to {1} by process id: {2}'.format( number, result, proc))

if __name__ == '__main__':
    numbers = [5, 10, 15, 20, 25]
    procs = []

    # 프로세스 중 하나에 에러가 생길 경우 그 프로세스만 중단된다. (에러처리하지 않을 경우)
    for index, number in enumerate(numbers):
        proc = Process(target=doubler, args=(number,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()