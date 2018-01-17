import numpy as np
import matplotlib.pyplot as plt

# 2차원 데이터 즉, 두 개의 실수 데이터 집합의 상관관계를 살펴보려면 scatter 명령으로 스캐터 플롯을 그린다. 스캐터 플롯의 점 하나의 위치는 데이터 하나의 x, y 값이다.


np.random.seed(0)
X = np.random.normal(0, 1, 100)
Y = np.random.normal(0, 1, 100)
plt.scatter(X, Y)
plt.show()




# 데이터가 2차원이 아니라 3차원 혹은 4차원인 경우에는 점 하나의 크기 혹은 색깔을 이용하여 다른 데이터 값을 나타낼 수도 있다. 이런 차트를 버블 차트(bubble chart)라고 한다. 크기는 s 인수로 색깔은 c 인수로 지정한다.


N = 30
np.random.seed(0)
x = np.random.rand(N)
y1 = np.random.rand(N)
y2 = np.random.rand(N)
y3 = np.pi * (15 * np.random.rand(N))**2
plt.scatter(x, y1, c=y2, s=y3)
plt.show()