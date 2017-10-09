import math

num_friends = [100,15,10,10,9,4,3,3,2,1]

def mean(x):
    return sum(x) / len(x)

def dot(v, w):
    return (v_i * w_i for v_i, w_i in zip (v,w))

def sum_of_squares(v):
    return sum(dot(v, v))

def de_mean(x): # 요소들과 평균의 차이
    x_bar = mean(x)
    print('mean',mean(x))
    return [x_i - x_bar for x_i in x]

def variance(x):
    n = len(x)
    deviatioins = de_mean(x)
    print(de_mean(x))
    return sum_of_squares(deviatioins) / (n - 1) # n으로 나누기보다 n-1로 나누어야 정확하게 보정됨 (위키참조)

print (variance(num_friends))