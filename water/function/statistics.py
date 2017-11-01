import math

def mean(x):
    return sum(x) / len(x)

def dot(v, w):
    return (v_i * w_i for v_i, w_i in zip (v,w))

def sum_of_squares(v):
    return sum(dot(v, v))

def de_mean(x): # 요소들과 평균의 차이
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def variance(x):
    n = len(x)
    deviatioins = de_mean(x)
    return sum_of_squares(deviatioins) / (n - 1) # n으로 나누기보다 n-1로 나누어야 정확하게 보정됨 (위키참조)

def standard_deviation(x):
    return math.sqrt(variance(x))

def covariance(x,y):
    n = len(x)
    return sum(dot(de_mean(x), de_mean(y))) / (n-1)

def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)

    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else :
        return 0 # 편차가 존재하지 않는다면 상관관계는 0









