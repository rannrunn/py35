import matplotlib.pyplot as plt
import numpy as  np

# 히스토그램을 그리기 위한 hist 명령도 있다. hist 명령은 bins 인수로 데이터를 집계할 구간 정보를 받는다. 반환값으로 데이터 집계 결과를 반환한다.

np.random.seed(0)
x = np.random.randn(1000)
arrays, bins, patches = plt.hist(x, bins=10)
plt.show()